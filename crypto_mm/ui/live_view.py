from __future__ import annotations

"""
TTY Live UI (single-window, ~500ms refresh).

Left panel  : Order book (top10 cumulative) + Spread/Mid/Microprice
              + Exec-spread KPIs + Ladder summary
              + STRATEGY: PnL R/U/T, Position/AvgPx/Exposure, Inventory sparkline, last simulated fills
Right panel : LAST 10 TRADES — aligned table: tsZ | type | side | qty | price
"""

import sys
import time
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, TextIO, Tuple
from collections import deque
from datetime import datetime, timezone
import math
import shutil


from ..core.types import utc_now_iso
from ..data.order_book import OrderBookL2
from ..data.trade_tape import TradeTape, TradeRecord
from ..mm.spread_kpi import KpiStore, spread_for_size

# ============================== Models ==========================================

SPARK_BARS = "▁▂▃▄▅▆▇█"


@dataclass(frozen=True)
class UiFill:
    ts_epoch: float
    side: str          # 'buy' | 'sell'
    px: float
    qty: float


@dataclass
class LiveState:
    """State passed to the renderer.

    Parameters
    ----------
    pair : str
    order_book : OrderBookL2
    trade_tape : TradeTape
    depth : int
    min_size : float
    kpi_store : KpiStore
    kpi_sizes : tuple[float, ...]

    Strategy state (live) — updated by the main loop:
    pnl_realized / pnl_unrealized / position_btc / avg_entry_px / exposure_usd
    inv_history (sparkline), fills (last simulated fills)
    """
    pair: str
    order_book: OrderBookL2
    trade_tape: TradeTape
    depth: int = 10
    min_size: float = 0.0
    kpi_store: KpiStore = field(default_factory=lambda: KpiStore(window_s=300))
    kpi_sizes: Tuple[float, ...] = (0.1, 1.0, 5.0, 10.0)

    # ---------- STRATEGY STATE ----------
    pnl_realized: float = 0.0
    pnl_unrealized: float = 0.0
    position_btc: float = 0.0
    avg_entry_px: Optional[float] = None
    exposure_usd: float = 0.0

    inv_history: Deque[float] = field(default_factory=lambda: deque(maxlen=60))
    fills: Deque[UiFill] = field(default_factory=lambda: deque(maxlen=8))


@dataclass
class RenderOptions:
    color: bool = True


# ============================ helpers ===========================================

def _fmt_qty(q: float) -> str:
    return f"{q:.6f}".rstrip("0").rstrip(".")


def _fmt_price(p: float) -> str:
    return f"{p:,.2f}"


def _goto(row: int, col: int = 1) -> str:
    return f"\x1b[{row};{col}H"


def _clear_line_right() -> str:
    return "\x1b[K"


def _c(code: str, s: str, enabled: bool) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if enabled else s


def _cumsum(levels: Sequence[Tuple[float, float]], n: int) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    s = 0.0
    for px, qty in levels[:n]:
        s += qty
        out.append((px, s))
    return out


def _iso(ts_epoch: float) -> str:
    return datetime.fromtimestamp(ts_epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def _sparkline(vals: Sequence[float], width: int) -> str:
    if not vals or width <= 0:
        return ""
    vmin = min(vals)
    vmax = max(vals)
    if not (vmax > vmin):
        return SPARK_BARS[len(SPARK_BARS) // 2] * min(width, len(vals))
    # simple downsample to match width
    step = max(1, len(vals) // width)
    sample = list(vals)[-step * width::step]
    out = []
    for v in sample[-width:]:
        t = (v - vmin) / (vmax - vmin)  # 0..1
        idx = min(len(SPARK_BARS) - 1, max(0, int(round(t * (len(SPARK_BARS) - 1)))))
        out.append(SPARK_BARS[idx])
    return "".join(out)


# ============================ Renderer ==========================================

class LiveRenderer:
    """Two panels with ANSI cursor addressing (single window updated in place)."""

    # Layout (1-based)
    LEFT_COL = 2
    WIDTH_TOTAL = 160         # canevas logique (borné à la largeur du terminal)
    LEFT_WIDTH = 100
    RIGHT_COL = LEFT_COL + LEFT_WIDTH + 4  # espace entre panneaux

    # Rows (left)
    ROW_HEADER = 1
    ROW_SEP1 = 2
    ROW_TITLES = 3
    ROW_TOP_HDR = 4
    ROW_TOP_START = 5          # 5..14
    ROW_TOP_END = 14
    ROW_METRICS = 15
    ROW_SEP2 = 16

    # --- KPIs spread ---
    ROW_SPREAD_KPI_HDR = 17        # "SPREAD KPI"
    ROW_SPREAD_KPI_COLHDR = 18     # entête colonnes
    ROW_SPREAD_KPI_START = 19      # 19..22 (4 lignes pour 0.1/1/5/10)
    ROW_SPREAD_KPI_END = 22

    # --- Microprice volatility (fenêtre fixe 50) ---
    ROW_MICROVOL_HDR = 23
    ROW_MICROVOL_LINE = 24

    ROW_LADDER_HDR = 25
    ROW_LADDER_BIDS = 26
    ROW_LADDER_ASKS = 27

    # ---- STRATEGY block under ladder ----
    ROW_STRAT_HDR = 28
    ROW_STRAT_PNL = 29
    ROW_STRAT_POS = 30
    ROW_STRAT_INV = 31
    ROW_FILLS_HDR = 32
    ROW_FILLS_START = 33        # 33..39
    ROW_FILLS_END = 39

    ROW_SEP3 = 40  # separator before right panel

    # Rows (right)
    ROW_RIGHT_TITLE = 3
    ROW_RIGHT_START = 4         # 4..21
    ROW_RIGHT_END = 21

    # Column widths (right)
    RIGHT_TS_W = 24
    RIGHT_TYPE_W = 9
    RIGHT_SIDE_W = 5
    RIGHT_QTY_W = 10
    RIGHT_PX_W = 14

    # Exchange tick pour l’affichage du spread (Kraken cash)
    TICK_SIZE = 0.10

    # --- KPI table col widths (garde un léger retrait = "colonne vide") ---
    KPI_INDENT = "  "           # colonne vide demandée
    KPI_SIZE_W = 6
    KPI_MED_W = 7
    KPI_NOBS_W = 6
    KPI_WIN_W = 8

    # Fenêtre fixe pour la vol microprice
    MICROVOL_N = 50

    def __init__(self, out: Optional[TextIO] = None, opts: Optional[RenderOptions] = None) -> None:
        self.out: TextIO = out if out is not None else sys.stdout
        self.opts = opts or RenderOptions()
        self.started = False
        from collections import deque
        # garde seulement les 50 derniers micro-prix
        self._micro_samples: Deque[float] = deque(maxlen=self.MICROVOL_N)

    # --------------- write primitive ---------------------
    def _w(self, s: str) -> None:
        try:
            self.out.write(s)
        except (ValueError, OSError):
            pass

    # --------------- geometry helpers --------------------
    def _right_remaining(self) -> int:
        """Remaining width for right panel given current terminal width."""
        import shutil
        try:
            cols = shutil.get_terminal_size(fallback=(self.WIDTH_TOTAL, 40)).columns
        except Exception:
            cols = self.WIDTH_TOTAL
        total_usable = min(cols, self.WIDTH_TOTAL)
        rem = total_usable - (self.RIGHT_COL - 1)
        return max(0, rem)

    # --------------- static drawing ----------------------
    def start(self, state: LiveState) -> None:
        if self.started:
            return
        # Clear + hide cursor
        self._w("\x1b[2J\x1b[H\x1b[?25l")

        # vertical separator
        for r in range(self.ROW_TITLES, self.ROW_SEP3 + 1):
            self._w(_goto(r, self.RIGHT_COL - 2) + "│")

        # headings
        self._w(_goto(self.ROW_SEP1, 1) + "=" * self.WIDTH_TOTAL + _clear_line_right())
        self._w(_goto(self.ROW_TITLES, self.LEFT_COL) +
                _c("36", "ORDER BOOK — TOP 10 LEVELS (cumulative sizes)", self.opts.color) +
                _clear_line_right())
        self._w(
            _goto(self.ROW_TITLES, self.RIGHT_COL)
            + _c("1", "LAST 10 TRADES", self.opts.color)
            + _clear_line_right()
        )

        # left sub-headers
        hdr_l = f"{_c('32', 'BID px', self.opts.color):>10}  {_c('32', 'cum_btc', self.opts.color):>10}"
        hdr_r = f"{_c('31', 'ASK px', self.opts.color):>10}  {_c('31', 'cum_btc', self.opts.color):>10}"
        self._w(_goto(self.ROW_TOP_HDR, self.LEFT_COL) + f"{hdr_l:<28}  ||  {hdr_r:<28}" + _clear_line_right())

        # Right table header
        right_hdr = (
            f"{'tsZ':<{self.RIGHT_TS_W}} "
            f"{'type':<{self.RIGHT_TYPE_W}} "
            f"{'side':<{self.RIGHT_SIDE_W}} "
            f"{'qty':>{self.RIGHT_QTY_W}} "
            f"{'price':>{self.RIGHT_PX_W}}"
        )
        rem = self._right_remaining()
        if rem > 0:
            self._w(_goto(self.ROW_RIGHT_TITLE, self.RIGHT_COL) + right_hdr[:rem] + _clear_line_right())

        # clear dynamic areas (left)
        for r in range(self.ROW_TOP_START, self.ROW_TOP_END + 1):
            self._w(_goto(r, self.LEFT_COL) + _clear_line_right())
        for r in (
            self.ROW_METRICS, self.ROW_SEP2,
            self.ROW_SPREAD_KPI_HDR, self.ROW_SPREAD_KPI_COLHDR,
            self.ROW_MICROVOL_HDR, self.ROW_MICROVOL_LINE
        ):
            self._w(_goto(r, 1) + _clear_line_right())
        for r in range(self.ROW_SPREAD_KPI_START, self.ROW_SPREAD_KPI_END + 1):
            self._w(_goto(r, self.LEFT_COL) + _clear_line_right())

        for r in (self.ROW_LADDER_HDR, self.ROW_LADDER_BIDS, self.ROW_LADDER_ASKS):
            self._w(_goto(r, self.LEFT_COL) + _clear_line_right())

        # Strategy area & fills
        for r in (self.ROW_STRAT_HDR, self.ROW_STRAT_PNL, self.ROW_STRAT_POS,
                  self.ROW_STRAT_INV, self.ROW_FILLS_HDR):
            self._w(_goto(r, 1) + _clear_line_right())
        for r in range(self.ROW_FILLS_START, self.ROW_FILLS_END + 1):
            self._w(_goto(r, self.LEFT_COL) + _clear_line_right())

        # right dynamic area
        for r in range(self.ROW_RIGHT_START, self.ROW_RIGHT_END + 1):
            self._w(_goto(r, self.RIGHT_COL) + _clear_line_right())

        # separator before right panel
        self._w(_goto(self.ROW_SEP3, 1) + "-" * self.WIDTH_TOTAL + _clear_line_right())

        try:
            self.out.flush()
        except Exception:
            pass

        self.started = True

    def stop(self) -> None:
        self._w("\x1b[?25h")
        try:
            self.out.flush()
        except Exception:
            pass

    # --------------- periodic update ---------------------
    def update(self, state: LiveState) -> None:
        import math, time
        ob = state.order_book
        tape = state.trade_tape

        # keep the gutter crisp after resizes
        for r in range(self.ROW_TITLES, self.ROW_SEP3 + 1):
            self._w(_goto(r, self.RIGHT_COL - 2) + "│")

        # header (pair + UTC now)
        self._w(_goto(self.ROW_HEADER, self.LEFT_COL) +
                _c("1", f"{state.pair}  @ {utc_now_iso()}", self.opts.color) + _clear_line_right())

        # left — top10 cumulative
        top = ob.top(50)
        bids = top["bids"]
        asks = top["asks"]
        cbids = _cumsum(bids, 10)
        casks = _cumsum(asks, 10)

        for i in range(10):
            row = self.ROW_TOP_START + i
            if i < len(cbids):
                bl = _c("32", f"{_fmt_price(cbids[i][0]):>10}  {_fmt_qty(cbids[i][1]):>10}", self.opts.color)
            else:
                bl = " " * 22
            if i < len(casks):
                ar = _c("31", f"{_fmt_price(casks[i][0]):>10}  {_fmt_qty(casks[i][1]):>10}", self.opts.color)
            else:
                ar = ""
            self._w(_goto(row, self.LEFT_COL) + f"{bl:<28}  ||  {ar:<28}"[:self.LEFT_WIDTH] + _clear_line_right())

        # left — metrics (spread/mid/micro) avec spread basé sur le tick
        best_bid = bids[0] if bids else None
        best_ask = asks[0] if asks else None
        if best_bid and best_ask:
            raw = best_ask[0] - best_bid[0]
            ticks = max(0, int(round(raw / self.TICK_SIZE)))
            spr_disp = ticks * self.TICK_SIZE
            spread = f"${_fmt_price(spr_disp)}  ({ticks} ticks)"
        else:
            spread = "—"
        mid_val = ob.mid_price()
        micro_val = ob.microprice()
        mid_s = _fmt_price(mid_val) if mid_val == mid_val else "—"
        micro_s = _fmt_price(micro_val) if micro_val == micro_val else "—"
        metrics = f"Spread: {spread}   |   Mid: {mid_s}   |   Microprice: {micro_s}"
        self._w(_goto(self.ROW_METRICS, self.LEFT_COL) + _c("36", metrics, self.opts.color) + _clear_line_right())

        # --- KPI spread (table size | median | n_obs | window_s) ---
        self._w(_goto(self.ROW_SEP2, 1) + "-" * self.WIDTH_TOTAL + _clear_line_right())
        self._w(_goto(self.ROW_SPREAD_KPI_HDR, self.LEFT_COL) + _c("1", "SPREAD KPI", self.opts.color) + _clear_line_right())

        # ligne d'entête des colonnes (alignée + colonne vide à gauche)
        colhdr = (
            f"{self.KPI_INDENT}"
            f"{'size':>{self.KPI_SIZE_W}} | "
            f"{'median':>{self.KPI_MED_W}} | "
            f"{'n_obs':>{self.KPI_NOBS_W}} | "
            f"{'window_s':>{self.KPI_WIN_W}}"
        )
        self._w(_goto(self.ROW_SPREAD_KPI_COLHDR, self.LEFT_COL) + colhdr[:self.LEFT_WIDTH] + _clear_line_right())

        # Alimenter la fenêtre (mesures instantanées)
        now = time.time()
        book = {"bids": bids, "asks": asks}
        for s in state.kpi_sizes:
            v = spread_for_size(book, s)
            v = None if (v is None or v < 0) else v  # ignore inversions
            state.kpi_store.append(now, s, v)

        agg = state.kpi_store.aggregates(now_epoch=now)
        cols = set(map(str, getattr(agg, "columns", [])))

        def _pick(*names: str) -> Optional[str]:
            for n in names:
                if n in cols:
                    return n
            return None

        col_med = _pick("median", "p50", "med")
        col_cnt = _pick("count", "n_obs", "n")

        def _cell_med(v: Optional[float]) -> str:
            return f"{v:>{self.KPI_MED_W}.2f}" if v is not None else " " * (self.KPI_MED_W - 1) + "—"
        def _cell_int(n: Optional[int]) -> str:
            return f"{n:>{self.KPI_NOBS_W}d}" if n is not None else f"{0:>{self.KPI_NOBS_W}d}"

        sizes = list(state.kpi_sizes)  # 4 tailles
        for i, s in enumerate(sizes):
            row = self.ROW_SPREAD_KPI_START + i
            med_val = None
            n_obs = 0
            if not getattr(agg, "empty", True):
                # clé robuste
                idx = None
                for cand in (float(s), s, str(s)):
                    try:
                        _ = agg.loc[cand]
                        idx = cand
                        break
                    except Exception:
                        continue
                if idx is not None:
                    r = agg.loc[idx]
                    if col_med and col_med in agg.columns:
                        try:
                            mv = float(r[col_med])
                            med_val = mv if (mv == mv) else None
                        except Exception:
                            med_val = None
                    if col_cnt and col_cnt in agg.columns:
                        try:
                            nv = int(r[col_cnt]) if r[col_cnt] == r[col_cnt] else 0
                        except Exception:
                            nv = 0
                        n_obs = nv

            line = (
                f"{self.KPI_INDENT}"
                f"{s:>{self.KPI_SIZE_W}g} | "
                f"{_cell_med(med_val)} | "
                f"{_cell_int(n_obs)} | "
                f"{state.kpi_store.window_s:>{self.KPI_WIN_W}d}"
            )
            self._w(_goto(row, self.LEFT_COL) + line[:self.LEFT_WIDTH] + _clear_line_right())

        # --- MICROPRICE VOL (σ over last 50 samples) ---
        self._w(_goto(self.ROW_MICROVOL_HDR, self.LEFT_COL) +
                _c("1", "MICRO VOL (last 50 samples)", self.opts.color) + _clear_line_right())

        # push current micro sample (fenêtre fixe N)
        if micro_val == micro_val:
            self._micro_samples.append(float(micro_val))

        def _std(vals: List[float]) -> Optional[float]:
            if len(vals) < 2:
                return None
            m = sum(vals) / len(vals)
            var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
            return math.sqrt(var)

        px_vals = list(self._micro_samples)
        px_std = _std(px_vals)

        rets: List[float] = []
        for i in range(1, len(px_vals)):
            p0, p1 = px_vals[i - 1], px_vals[i]
            if p0 > 0:
                rets.append((p1 / p0) - 1.0)
        ret_std_bps = (_std(rets) * 10_000.0) if len(rets) >= 2 else None

        n_samp = len(px_vals)
        vol_line = (
            f"σ_px=${px_std:,.2f}" if px_std is not None else "σ_px=—"
        ) + "   " + (
            f"σ_ret={ret_std_bps:,.1f} bps" if ret_std_bps is not None else "σ_ret=—"
        ) + f"   samples={n_samp:d}   window_n={self.MICROVOL_N:d}"
        self._w(_goto(self.ROW_MICROVOL_LINE, self.LEFT_COL) + vol_line[:self.LEFT_WIDTH] + _clear_line_right())

        # left — ladder summary (with clearer ranges + $ depth)
        b_lvls = [(p, q) for (p, q) in bids[:state.depth] if q >= state.min_size]
        a_lvls = [(p, q) for (p, q) in asks[:state.depth] if q >= state.min_size]
        b_sum = sum(q for _, q in b_lvls)
        a_sum = sum(q for _, q in a_lvls)
        # USD depth approx using mid
        mid_px = mid_val if mid_val == mid_val else (a_lvls[0][0] if a_lvls else (b_lvls[0][0] if b_lvls else 0.0))
        b_depth_usd = b_sum * mid_px
        a_depth_usd = a_sum * mid_px
        b_rng = f"{int(b_lvls[-1][0])}..{int(b_lvls[0][0])}" if b_lvls else "—"
        a_rng = f"{int(a_lvls[0][0])}..{int(a_lvls[-1][0])}" if a_lvls else "—"

        self._w(_goto(self.ROW_LADDER_HDR, self.LEFT_COL) +
                _c("1", f"LADDER (min_size={state.min_size:g}, depth={state.depth})", self.opts.color) +
                _clear_line_right())
        self._w(_goto(self.ROW_LADDER_BIDS, self.LEFT_COL) +
                (f"BIDS ≥{state.min_size:g}: lvls={len(b_lvls):<2}  "
                 f"sum={_fmt_qty(b_sum)} BTC  |  depth≈ ${b_depth_usd:,.2f}  |  range {b_rng}")[:self.LEFT_WIDTH] +
                _clear_line_right())
        self._w(_goto(self.ROW_LADDER_ASKS, self.LEFT_COL) +
                (f"ASKS ≥{state.min_size:g}: lvls={len(a_lvls):<2}  "
                 f"sum={_fmt_qty(a_sum)} BTC  |  depth≈ ${a_depth_usd:,.2f}  |  range {a_rng}")[:self.LEFT_WIDTH] +
                _clear_line_right())

        # -------------------- STRATEGY STATE (under ladder) --------------------
        pnl_r = state.pnl_realized
        pnl_u = state.pnl_unrealized
        pnl_t = pnl_r + pnl_u

        def _clr_sign(v: float) -> str:
            if not self.opts.color:
                return f"{v:,.2f}"
            if v > 1e-12:
                return _c("32", f"+{v:,.2f}", True)   # green
            if v < -1e-12:
                return _c("31", f"{v:,.2f}", True)    # red
            return f"{v:,.2f}"

        self._w(_goto(self.ROW_STRAT_HDR, self.LEFT_COL) +
                _c("1", "STRATEGY — PnL / Position / Inventory / Sim Fills", self.opts.color) +
                _clear_line_right())

        pnl_line = f"PnL R/U/T: { _clr_sign(pnl_r) } / { _clr_sign(pnl_u) } / { _clr_sign(pnl_t) }  USD"
        self._w(_goto(self.ROW_STRAT_PNL, self.LEFT_COL) + pnl_line[:self.LEFT_WIDTH] + _clear_line_right())

        avgpx = "—" if state.avg_entry_px is None or not (state.avg_entry_px == state.avg_entry_px) else _fmt_price(state.avg_entry_px)
        pos_line = (
            f"POS: {_fmt_qty(state.position_btc)} BTC  |  AvgPx: {avgpx}  |  Exposure: {_fmt_price(state.exposure_usd)} USD"
        )
        self._w(_goto(self.ROW_STRAT_POS, self.LEFT_COL) + pos_line[:self.LEFT_WIDTH] + _clear_line_right())

        # Inventory sparkline
        hist = list(state.inv_history)
        if hist:
            span = (min(hist), max(hist))
            spark_w = max(40, min(60, self.LEFT_WIDTH - 40))
            SPARK_BARS = "▁▂▃▄▅▆▇█"
            if span[1] > span[0]:
                def _to_bar(v: float) -> str:
                    t = (v - span[0]) / (span[1] - span[0])
                    idx = min(len(SPARK_BARS) - 1, max(0, int(round(t * (len(SPARK_BARS) - 1)))))
                    return SPARK_BARS[idx]
                step = max(1, len(hist) // spark_w)
                sample = hist[-step * spark_w::step][-spark_w:]
                spark = "".join(_to_bar(v) for v in sample)
            else:
                spark = "▅" * spark_w
            inv_legend = f"INV: last={_fmt_qty(hist[-1])} BTC  [{_fmt_qty(span[0])} .. {_fmt_qty(span[1])}]"
            self._w(_goto(self.ROW_STRAT_INV, self.LEFT_COL) + (inv_legend + "  " + spark)[:self.LEFT_WIDTH] + _clear_line_right())
        else:
            self._w(_goto(self.ROW_STRAT_INV, self.LEFT_COL) + "INV: —" + _clear_line_right())

        # Fills header
        self._w(_goto(self.ROW_FILLS_HDR, self.LEFT_COL) + _c("1", "SIM FILLS (last)", self.opts.color) + _clear_line_right())

        # Last simulated fills
        nrows = self.ROW_FILLS_END - self.ROW_FILLS_START + 1
        fills_list: List[Optional[UiFill]] = list(state.fills)[-nrows:]
        if len(fills_list) < nrows:
            fills_list = [None] * (nrows - len(fills_list)) + fills_list  # top padding

        for i, f in enumerate(fills_list):
            row = self.ROW_FILLS_START + i
            if f is None:
                self._w(_goto(row, self.LEFT_COL) + _clear_line_right())
                continue
            tsZ = _iso(f.ts_epoch)
            side_txt = "buy" if f.side == "buy" else "sell"
            if self.opts.color:
                side_txt = _c("32", "buy", True) if f.side == "buy" else _c("31", "sell", True)
            qty_s = _fmt_qty(f.qty)
            px_s = _fmt_price(f.px)
            line = f"{tsZ}  {side_txt:<5}  qty={qty_s:>10}  px={px_s:>12}"
            self._w(_goto(row, self.LEFT_COL) + line[:self.LEFT_WIDTH] + _clear_line_right())

        # =================== RIGHT PANEL: LAST TRADES ===========================
        rem = self._right_remaining()
        if rem > 0:
            hdr = (
                f"{'tsZ':<{self.RIGHT_TS_W}} "
                f"{'type':<{self.RIGHT_TYPE_W}} "
                f"{'side':<{self.RIGHT_SIDE_W}} "
                f"{'qty':>{self.RIGHT_QTY_W}} "
                f"{'price':>{self.RIGHT_PX_W}}"
            )[:rem]
            self._w(_goto(self.ROW_RIGHT_TITLE, self.RIGHT_COL) + hdr + _clear_line_right())

            nrows_right = self.ROW_RIGHT_END - self.ROW_RIGHT_START + 1
            trades: List[TradeRecord] = tape.last(nrows_right)
            if len(trades) < nrows_right:
                lines: List[Optional[TradeRecord]] = [None] * (nrows_right - len(trades)) + trades
            else:
                lines = trades[-nrows_right:]  # oldest .. newest

            for i, tr in enumerate(lines):
                row = self.ROW_RIGHT_START + i
                if tr is None:
                    self._w(_goto(row, self.RIGHT_COL) + _clear_line_right())
                    continue
                tsZ = _iso(tr.ts_epoch)
                typ = "trade"
                side_txt = tr.side
                qty_s = _fmt_qty(tr.qty_btc)
                px_s = _fmt_price(tr.price_usd_per_btc)
                line = (
                    f"{tsZ:<{self.RIGHT_TS_W}} "
                    f"{typ:<{self.RIGHT_TYPE_W}} "
                    f"{side_txt:<{self.RIGHT_SIDE_W}} "
                    f"{qty_s:>{self.RIGHT_QTY_W}} "
                    f"{px_s:>{self.RIGHT_PX_W}}"
                )[:rem]
                self._w(_goto(row, self.RIGHT_COL) + line + _clear_line_right())

        try:
            self.out.flush()
        except Exception:
            pass

_RENDERER: Optional[LiveRenderer] = None


def render(state: LiveState, *, out: Optional[TextIO] = None, opts: Optional[RenderOptions] = None) -> None:
    """Render/update the live UI."""
    global _RENDERER
    if _RENDERER is None or (out is not None and _RENDERER.out is not out):
        _RENDERER = LiveRenderer(out=out, opts=opts)
        _RENDERER.start(state)
    else:
        if opts is not None:
            _RENDERER.opts = opts
    _RENDERER.update(state)


def stop_render() -> None:
    """Restore cursor visibility at the end."""
    global _RENDERER
    if _RENDERER is not None:
        _RENDERER.stop()
