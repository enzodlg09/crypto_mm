from __future__ import annotations

"""
TTY Live UI (single-window, 500ms refresh).

Left panel  : Order book (top10 cumulative) + Spread/Mid/Microprice
              + Exec-spread KPIs + Ladder summary (BIDS/ASKS)
Right panel : Events (snapshot / trades / resync) — aligned table:
              tsZ | type | side | qty | price
"""

import sys
import time
import re
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, TextIO, Tuple
from collections import deque
from datetime import datetime, timezone


from ..core.types import utc_now_iso
from ..data.order_book import OrderBookL2
from ..data.trade_tape import TradeTape, TradeRecord
from ..mm.spread_kpi import KpiStore, spread_for_size


# ============================== Models ==========================================

@dataclass
class LiveState:
    """Container passé au renderer."""
    pair: str
    order_book: OrderBookL2
    trade_tape: TradeTape
    depth: int = 10
    min_size: float = 0.0
    kpi_store: KpiStore = field(default_factory=lambda: KpiStore(window_s=300))
    kpi_sizes: Tuple[float, ...] = (0.1, 1.0, 5.0, 10.0)


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


# ============================ Events ============================================

@dataclass
class _EventRow:
    ts: str
    etype: str   # snapshot | trade | resync | info
    side: str = ""
    qty: str = ""
    price: str = ""


# ============================ Renderer ==========================================

class LiveRenderer:
    """Deux panneaux avec adressage ANSI (une seule fenêtre qui se met à jour)."""

    # Layout (1-based)
    LEFT_COL = 2
    WIDTH_TOTAL = 120
    LEFT_WIDTH = 78
    RIGHT_COL = LEFT_COL + LEFT_WIDTH + 2  # séparateur vertical à RIGHT_COL-2

    # Rows (gauche)
    ROW_HEADER = 1
    ROW_SEP1 = 2
    ROW_TITLES = 3
    ROW_TOP_HDR = 4
    ROW_TOP_START = 5          # 5..14
    ROW_TOP_END = 14
    ROW_METRICS = 15
    ROW_SEP2 = 16
    ROW_KPI1 = 17
    ROW_KPI2 = 18
    ROW_LADDER_HDR = 19
    ROW_LADDER_BIDS = 20
    ROW_LADDER_ASKS = 21
    ROW_SEP3 = 22

    # Rows (droite)
    ROW_RIGHT_TITLE = 3
    ROW_RIGHT_START = 4        # 4..21
    ROW_RIGHT_END = 21

    # Col sizes (droite)
    RIGHT_TS_W = 24
    RIGHT_TYPE_W = 9
    RIGHT_SIDE_W = 5
    RIGHT_QTY_W = 10
    RIGHT_PX_W = 14

    # "trade buy 0.001 @ 116,000.00"
    _TRADE_RE = re.compile(
        r"^trade\s+(buy|sell)\s+([0-9]*\.?[0-9]+)\s*@\s*([0-9,]*\.?[0-9]+)$",
        re.IGNORECASE,
    )

    def __init__(self, out: Optional[TextIO] = None, opts: Optional[RenderOptions] = None) -> None:
        self.out: TextIO = out if out is not None else sys.stdout
        self.opts = opts or RenderOptions()
        self.started = False
        self.events: Deque[_EventRow] = deque(maxlen=(self.ROW_RIGHT_END - self.ROW_RIGHT_START + 1))
        self._last_trade_event_id: Optional[int] = None
        self._last_seen_trade: Optional[TradeRecord] = None

    # --------------- primitives d'écriture ---------------
    def _w(self, s: str) -> None:
        try:
            self.out.write(s)
        except (ValueError, OSError):
            pass

    # --------------- API évènements ----------------------
    def add_event(self, msg: str) -> None:
        """Append a parsed/normalized event row.

        - Accepte le format structuré :  "trade <side> <qty> @ <price>"
        - Tolère les messages "timestamp ... trad/snap" (génériques),
          en réémettant un trade détaillé à partir de la tape si possible.
        """
        ts = utc_now_iso()
        raw = msg.strip()
        low = raw.lower()

        # 1) Format structuré
        m = self._TRADE_RE.match(raw)
        if m:
            side = m.group(1).lower()
            qty = _fmt_qty(float(m.group(2)))
            px = _fmt_price(float(m.group(3).replace(",", "")))
            self.events.append(_EventRow(ts=ts, etype="trade", side=side, qty=qty, price=px))
            return

        # 2) Messages génériques "timestamp ... trad" / "timestamp ... snap"
        #    -> on tente de reconstruire une ligne de trade depuis la tape
        ends_with_trad = low.rstrip().endswith(" trad") or low.rstrip().endswith(" trade")
        ends_with_snap = low.rstrip().endswith(" snap") or " snapshot" in low

        if ends_with_trad:
            if self._last_seen_trade is not None:
                tr = self._last_seen_trade
                self.events.append(
                    _EventRow(
                        ts=ts,
                        etype="trade",
                        side=tr.side,
                        qty=_fmt_qty(tr.qty_btc),
                        price=_fmt_price(tr.price_usd_per_btc),
                    )
                )
            # Sinon, on ignore la ligne "trad" vide (pas d'infos utiles)
            return

        if ends_with_snap:
            self.events.append(_EventRow(ts=ts, etype="snapshot"))
            return

        # 3) Autres messages : on peut soit ignorer, soit loguer en 'info'
        # Pour éviter de polluer la table, on ignore par défaut.
        return

    # --------------- dessin statique ---------------------
    def start(self, state: LiveState) -> None:
        if self.started:
            return
        # Clear + hide cursor
        self._w("\x1b[2J\x1b[H\x1b[?25l")

        # Séparateur vertical
        for r in range(self.ROW_TITLES, self.ROW_SEP3 + 1):
            self._w(_goto(r, self.RIGHT_COL - 2) + "│")

        # Entêtes
        self._w(_goto(self.ROW_SEP1, 1) + "=" * self.WIDTH_TOTAL + _clear_line_right())
        self._w(_goto(self.ROW_TITLES, self.LEFT_COL) +
                _c("36", "ORDER BOOK — TOP 10 LEVELS (cumulative sizes)", self.opts.color) +
                _clear_line_right())
        self._w(
            _goto(self.ROW_TITLES, self.RIGHT_COL)
            + _c("1", "LAST 10 TRADES (tsZ   type  side   qty     price)", self.opts.color)
            + _clear_line_right()
        )

        # Sous-entêtes book
        hdr_l = f"{_c('32', 'BID px', self.opts.color):>10}  {_c('32', 'cum_btc', self.opts.color):>10}"
        hdr_r = f"{_c('31', 'ASK px', self.opts.color):>10}  {_c('31', 'cum_btc', self.opts.color):>10}"
        self._w(_goto(self.ROW_TOP_HDR, self.LEFT_COL) + f"{hdr_l:<28}  ||  {hdr_r:<28}" + _clear_line_right())

        # Entête colonne droite
        right_hdr = (
            f"{'tsZ':<{self.RIGHT_TS_W}} "
            f"{'type':<{self.RIGHT_TYPE_W}} "
            f"{'side':<{self.RIGHT_SIDE_W}} "
            f"{'qty':>{self.RIGHT_QTY_W}} "
            f"{'price':>{self.RIGHT_PX_W}}"
        )
        self._w(_goto(self.ROW_RIGHT_TITLE, self.RIGHT_COL) + right_hdr + _clear_line_right())

        # Nettoyage des zones dynamiques
        for r in range(self.ROW_TOP_START, self.ROW_TOP_END + 1):
            self._w(_goto(r, self.LEFT_COL) + _clear_line_right())
        for r in (self.ROW_METRICS, self.ROW_SEP2, self.ROW_KPI1, self.ROW_KPI2,
                  self.ROW_LADDER_HDR, self.ROW_LADDER_BIDS, self.ROW_LADDER_ASKS, self.ROW_SEP3):
            self._w(_goto(r, 1) + _clear_line_right())
        for r in range(self.ROW_RIGHT_START, self.ROW_RIGHT_END + 1):
            self._w(_goto(r, self.RIGHT_COL) + _clear_line_right())

        self.out.flush()
        self.started = True

    def stop(self) -> None:
        self._w("\x1b[?25h")
        try:
            self.out.flush()
        except Exception:
            pass

    # --------------- mise à jour périodique --------------
    def update(self, state: LiveState) -> None:
        ob = state.order_book
        tape = state.trade_tape

        # Header (pair + UTC now)
        self._w(_goto(self.ROW_HEADER, self.LEFT_COL) +
                _c("1", f"{state.pair}  @ {utc_now_iso()}", self.opts.color) + _clear_line_right())

        # Top10 cumulé
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

        # Spread / Mid / Microprice
        best_bid = bids[0] if bids else None
        best_ask = asks[0] if asks else None
        if best_bid and best_ask:
            spr = best_ask[0] - best_bid[0]
            spread = f"${_fmt_price(spr)}  ({int(spr/1.0)} ticks)"
        else:
            spread = "—"
        mid = ob.mid_price()
        micro = ob.microprice()
        mid_s = _fmt_price(mid) if mid == mid else "—"
        micro_s = _fmt_price(micro) if micro == micro else "—"
        metrics = f"Spread: {spread}   |   Mid: {mid_s}   |   Microprice: {micro_s}"
        self._w(_goto(self.ROW_METRICS, self.LEFT_COL) + _c("36", metrics, self.opts.color) + _clear_line_right())

        # KPIs: spreads exécutables
        now = time.time()
        book = {"bids": bids, "asks": asks}
        inst_vals: List[Optional[float]] = []
        for s in state.kpi_sizes:
            v = spread_for_size(book, s)
            state.kpi_store.append(now, s, v)
            inst_vals.append(v)

        agg = state.kpi_store.aggregates(now_epoch=now)
        med = {float(k): float(agg.loc[k, "median"]) for k in agg.index} if not agg.empty else {}

        def _fmt(v: Optional[float]) -> str:
            return "—" if v is None else f"${v:,.2f}"

        # Lignes compactes → tiennent dans LEFT_WIDTH
        inst_line = " | ".join(f"{s:g}: {_fmt(v)}" for s, v in zip(state.kpi_sizes, inst_vals))
        med_line = " | ".join(f"{s:g}: {_fmt(med.get(float(s)))}" for s in state.kpi_sizes)

        self._w(_goto(self.ROW_SEP2, 1) + "-" * self.WIDTH_TOTAL + _clear_line_right())
        self._w(_goto(self.ROW_KPI1, self.LEFT_COL) + _c("1", "EXEC SPREADS (instant)  ", self.opts.color) +
                inst_line[:self.LEFT_WIDTH] + _clear_line_right())
        self._w(_goto(self.ROW_KPI2, self.LEFT_COL) + _c("1", "EXEC SPREADS (5m median) ", self.opts.color) +
                med_line[:self.LEFT_WIDTH] + _clear_line_right())

        # Ladder propre (2 lignes)
        b_lvls = [(p, q) for (p, q) in bids if q >= state.min_size]
        a_lvls = [(p, q) for (p, q) in asks if q >= state.min_size]
        b_sum = sum(q for _, q in b_lvls)
        a_sum = sum(q for _, q in a_lvls)
        b_rng = f"{int(b_lvls[-1][0])}..{int(b_lvls[0][0])}" if b_lvls else "—"
        a_rng = f"{int(a_lvls[0][0])}..{int(a_lvls[-1][0])}" if a_lvls else "—"

        self._w(_goto(self.ROW_LADDER_HDR, self.LEFT_COL) +
                _c("1", f"LADDER (min_size={state.min_size:g}, depth={state.depth})", self.opts.color) +
                _clear_line_right())
        self._w(_goto(self.ROW_LADDER_BIDS, self.LEFT_COL) +
                (f"BIDS ≥{state.min_size:g}: lvls={len(b_lvls):<3}  sum={_fmt_qty(b_sum)} BTC  |  range {b_rng}")[:self.LEFT_WIDTH] +
                _clear_line_right())
        self._w(_goto(self.ROW_LADDER_ASKS, self.LEFT_COL) +
                (f"ASKS ≥{state.min_size:g}: lvls={len(a_lvls):<3}  sum={_fmt_qty(a_sum)} BTC  |  range {a_rng}")[:self.LEFT_WIDTH] +
                _clear_line_right())

        # Séparateur avant panneau droit
        self._w(_goto(self.ROW_SEP3, 1) + "-" * self.WIDTH_TOTAL + _clear_line_right())

        # ====== Right panel: LAST TRADES (side / qty / price) ======
        width = self.WIDTH_TOTAL - self.RIGHT_COL - 1

        # On prend les N derniers trades de la tape (nouveaux en bas)
        nrows = self.ROW_RIGHT_END - self.ROW_RIGHT_START + 1
        trades = state.trade_tape.last(nrows)
        pad = nrows - len(trades)

        # Lignes vides (en haut) si peu de trades
        for i in range(pad):
            row = self.ROW_RIGHT_START + i
            self._w(_goto(row, self.RIGHT_COL) + _clear_line_right())

        # Rendu des trades (du plus ancien au plus récent)
        for i, tr in enumerate(trades[-nrows:]):
            row = self.ROW_RIGHT_START + pad + i

            # tsZ : si on a un epoch, on le formate; sinon on garde tr.ts
            tsZ: str
            if hasattr(tr, "ts_epoch") and tr.ts_epoch is not None:
                tsZ = datetime.fromtimestamp(float(tr.ts_epoch), tz=timezone.utc).isoformat(timespec="seconds").replace(
                    "+00:00", "Z")
            else:
                tsZ = getattr(tr, "ts", "") or utc_now_iso()

            typ = _c("35", "trade", self.opts.color)  # violet
            side_col = _c("32", "buy", self.opts.color) if tr.side == "buy" else _c("31", "sell", self.opts.color)
            qty_s = _fmt_qty(float(getattr(tr, "qty_btc", 0.0)))
            px_s = _fmt_price(float(getattr(tr, "price_usd_per_btc", 0.0)))

            line = (
                f"{tsZ:<{self.RIGHT_TS_W}} "
                f"{typ:<{self.RIGHT_TYPE_W}} "
                f"{side_col:<{self.RIGHT_SIDE_W}} "
                f"{qty_s:>{self.RIGHT_QTY_W}} "
                f"{px_s:>{self.RIGHT_PX_W}}"
            )
            self._w(_goto(row, self.RIGHT_COL) + line[:width] + _clear_line_right())
        # Remember latest trade (for converting generic 'trad' lines)

        try:
            self.out.flush()
        except Exception:
            pass


# ============================ Façade module =====================================

_RENDERER: Optional[LiveRenderer] = None


def render(state: LiveState, *, out: Optional[TextIO] = None, opts: Optional[RenderOptions] = None) -> None:
    """Render/update la live UI."""
    global _RENDERER
    if _RENDERER is None or (out is not None and _RENDERER.out is not out):
        _RENDERER = LiveRenderer(out=out, opts=opts)
        _RENDERER.start(state)
    else:
        if opts is not None:
            _RENDERER.opts = opts
    _RENDERER.update(state)


def stop_render() -> None:
    """Restore le curseur à la fin."""
    global _RENDERER
    if _RENDERER is not None:
        _RENDERER.stop()


def log_event(msg: str) -> None:
    """Append un évènement dans le panneau de droite."""
    global _RENDERER
    if _RENDERER is not None:
        _RENDERER.add_event(msg)
