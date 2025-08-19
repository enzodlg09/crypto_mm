from __future__ import annotations

import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, TextIO, Tuple

from ..core.types import utc_now_iso
from ..data.order_book import OrderBookL2
from ..data.trade_tape import TradeTape, TradeRecord
from ..mm.spread_kpi import KpiStore, spread_for_size
from ..mm.fair_price import FairPrice


@dataclass
class LiveState:
    """State container for live rendering."""
    pair: str
    order_book: OrderBookL2
    trade_tape: TradeTape
    depth: int = 10
    min_size: float = 0.0
    # KPI spreads + fair price (gauche)
    kpi_store: KpiStore = KpiStore(window_s=300)
    kpi_sizes: Tuple[float, ...] = (0.1, 1.0, 5.0, 10.0)
    fair: FairPrice = FairPrice(mode="microprice", alpha=0.5)
    # Fair prices spécifiques au panneau events (ewma a besoin d'état persistant)
    fair_events: Dict[str, FairPrice] = field(default_factory=dict)


@dataclass
class RenderOptions:
    """Rendering options."""
    show_events: bool = False   # if False -> show last trades on right
    color: bool = True          # ANSI colors on/off
    ladder_mode: str = "summary"  # 'summary' | 'levels' | 'off'
    ladder_levels: int = 6        # used when ladder_mode == 'levels'
    # --- Nouveaux réglages pour fair prices dans le panneau Events ---
    events_fp_modes: Tuple[str, ...] = ()  # ex: ("mid","micro","ewma")
    fp_alpha: float = 0.5                   # microprice alpha
    fp_lambda: float = 1.0                  # ewma lambda (/s)


# ---------- small formatting helpers ----------

def _format_levels_cumsum(levels: List[Tuple[float, float]], n: int) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    s = 0.0
    for (p, q) in levels[:n]:
        s += q
        out.append((p, s))
    return out


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


# ---------- renderer ----------

@dataclass
class _EventRow:
    ts: str
    etype: str
    msg: str
    side: Optional[str] = None
    qty: Optional[float] = None
    price: Optional[float] = None


class LiveRenderer:
    """
    Two-panel single-window renderer using ANSI cursor addressing.
    Left: OrderBook (top10, metrics, ladder/KPIs)
    Right: Event log OR Last trades (+ fair prices si demandé)
    """

    # Layout (1-based)
    LEFT_COL = 2
    WIDTH_TOTAL = 120
    LEFT_WIDTH = 78
    RIGHT_COL = LEFT_COL + LEFT_WIDTH + 2  # vertical separator at RIGHT_COL-2

    # Rows
    ROW_HEADER = 1
    ROW_SEP1 = 2

    ROW_TITLES = 3
    ROW_TOP_HDR = 4
    ROW_TOP_START = 5         # rows 5..14
    ROW_TOP_END = 14

    ROW_METRICS = 15
    ROW_SEP2 = 16

    ROW_L1 = 17               # nous réutilisons L1/L2/L3 pour KPI + fair price
    ROW_L2 = 18
    ROW_L3 = 19

    ROW_SEP3 = 20

    ROW_RIGHT_TITLE = 3
    ROW_RIGHT_START = 4       # rows 4..19
    ROW_RIGHT_END = 19

    # Right panel column widths
    RIGHT_TS_W = 24
    RIGHT_TYPE_W = 9
    RIGHT_SIDE_W = 5
    RIGHT_QTY_W = 10
    RIGHT_PX_W = 14

    # Regex pour parser "trade buy 0.001 @ 116,000.00"
    _TRADE_RE = re.compile(
        r"^trade\s+(buy|sell)\s+([0-9]*\.?[0-9]+)\s*@\s*([0-9,]*\.?[0-9]+)",
        re.IGNORECASE,
    )

    def __init__(self, out: Optional[TextIO] = None, opts: Optional[RenderOptions] = None) -> None:
        self.out: TextIO = out if out is not None else sys.stdout
        self.started: bool = False
        self.events: Deque[_EventRow] = deque(maxlen=(self.ROW_RIGHT_END - self.ROW_RIGHT_START + 1))
        self.opts: RenderOptions = opts or RenderOptions()
        self._last_trade_event_id: Optional[int] = None

    def _w(self, s: str) -> None:
        try:
            self.out.write(s)
        except (ValueError, OSError):
            pass

    def start(self, state: LiveState) -> None:
        if self.started:
            return
        # Clear screen, go home, hide cursor
        self._w("\x1b[2J\x1b[H\x1b[?25l")

        # Vertical separator
        for r in range(self.ROW_TITLES, self.ROW_SEP3 + 1):
            self._w(_goto(r, self.RIGHT_COL - 2) + "│")

        # Top bar
        self._w(_goto(self.ROW_SEP1, 1) + "=" * self.WIDTH_TOTAL + _clear_line_right())
        title_left = _c("36", "ORDER BOOK — TOP 10 (cumulative sizes)", self.opts.color)  # cyan
        self._w(_goto(self.ROW_TITLES, self.LEFT_COL) + f"{title_left}" + _clear_line_right())
        self._w(_goto(self.ROW_TITLES, self.RIGHT_COL) + _clear_line_right())

        # LHS headers
        hdr_l = f"{_c('32', 'BID px', self.opts.color):>10}  {_c('32', 'cum_btc', self.opts.color):>10}"  # green
        hdr_r = f"{_c('31', 'ASK px', self.opts.color):>10}  {_c('31', 'cum_btc', self.opts.color):>10}"  # red
        self._w(_goto(self.ROW_TOP_HDR, self.LEFT_COL) + f"{hdr_l:<28}  ||  {hdr_r:<28}" + _clear_line_right())

        # Clear dynamic areas
        for r in range(self.ROW_TOP_START, self.ROW_TOP_END + 1):
            self._w(_goto(r, self.LEFT_COL) + _clear_line_right())
        self._w(_goto(self.ROW_METRICS, self.LEFT_COL) + _clear_line_right())
        self._w(_goto(self.ROW_SEP2, 1) + "-" * self.WIDTH_TOTAL + _clear_line_right())
        for r in (self.ROW_L1, self.ROW_L2, self.ROW_L3):
            self._w(_goto(r, self.LEFT_COL) + _clear_line_right())
        self._w(_goto(self.ROW_SEP3, 1) + "-" * self.WIDTH_TOTAL + _clear_line_right())

        # Right panel clear area
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

    # ---------- Event log API ----------
    def add_event(self, msg: str) -> None:
        """
        Append an event as (_EventRow). For trade messages, also extract side/qty/price.
        Accepted trade text formats include: 'trade buy 0.0123 @ 116,000.12'
        """
        ts = utc_now_iso()
        raw = msg.replace("\n", " ").strip()

        m = self._TRADE_RE.match(raw)
        if m:
            side = m.group(1).lower()
            qty = float(m.group(2))
            px = float(m.group(3).replace(",", ""))
            self.events.append(_EventRow(ts=ts, etype="trade", msg="", side=side, qty=qty, price=px))
            return

        mlow = raw.lower()
        if mlow.startswith("snapshot"):
            self.events.append(_EventRow(ts=ts, etype="snapshot", msg=raw))
        elif "resync" in mlow:
            self.events.append(_EventRow(ts=ts, etype="resync", msg=raw))
        else:
            self.events.append(_EventRow(ts=ts, etype="info", msg=raw))

    # ---------- Dynamic updates ----------
    def update(self, state: LiveState) -> None:
        ob = state.order_book
        tape = state.trade_tape

        # Header: pair + ts
        header = f"{state.pair}  @ {utc_now_iso()}"
        self._w(_goto(self.ROW_HEADER, self.LEFT_COL) + _c("1", header, self.opts.color) + _clear_line_right())

        # Top-10 (cumulative)
        top = ob.top(50)
        bids = top["bids"]
        asks = top["asks"]
        cbids = _format_levels_cumsum(bids, 10)
        casks = _format_levels_cumsum(asks, 10)

        for i in range(10):
            row = self.ROW_TOP_START + i
            if i < len(cbids):
                bl = f"{_fmt_price(cbids[i][0]):>10}  {_fmt_qty(cbids[i][1]):>10}"
                bl = _c("32", bl, self.opts.color)  # green
            else:
                bl = " " * 22
            if i < len(casks):
                ar = f"{_fmt_price(casks[i][0]):>10}  {_fmt_qty(casks[i][1]):>10}"
                ar = _c("31", ar, self.opts.color)  # red
            else:
                ar = ""
            line = f"{bl:<28}  ||  {ar:<28}"
            self._w(_goto(row, self.LEFT_COL) + line[:self.LEFT_WIDTH] + _clear_line_right())

        # Spread / mid / microprice
        best_bid = bids[0] if bids else None
        best_ask = asks[0] if asks else None
        if best_bid and best_ask:
            spread = best_ask[0] - best_bid[0]
            ticks = int(spread / 1.0)
            spread_str = f"${_fmt_price(spread)}  ({ticks} ticks)"
        else:
            spread_str = "—"

        mid = ob.mid_price()
        micro = ob.microprice()
        mid_str = _fmt_price(mid) if mid == mid else "—"
        micro_str = _fmt_price(micro) if micro == micro else "—"
        metrics = f"Spread: {spread_str}   |   Mid: {mid_str}   |   Microprice: {micro_str}"
        self._w(_goto(self.ROW_METRICS, self.LEFT_COL) + _c("36", metrics, self.opts.color) + _clear_line_right())

        # --- KPI spreads exécutables @ tailles fixes + Fair price (gauche) ---
        book = {"bids": bids, "asks": asks}
        now = time.time()
        inst_spreads = []
        for s in state.kpi_sizes:
            spr = spread_for_size(book, s)
            state.kpi_store.append(now, s, spr)
            inst_spreads.append(spr)

        agg = state.kpi_store.aggregates(now_epoch=now)
        med_map = {float(idx): float(agg.loc[idx, "median"]) for idx in agg.index} if not agg.empty else {}

        def _fmt_s(x: Optional[float]) -> str:
            return "—" if x is None else f"${x:,.2f}"

        inst_line = " | ".join(f"{s:g} BTC: {_fmt_s(v)}" for s, v in zip(state.kpi_sizes, inst_spreads))
        med_line = " | ".join(f"{s:g} BTC: {_fmt_s(med_map.get(float(s)))}" for s in state.kpi_sizes)

        self._w(_goto(self.ROW_SEP2, 1) + "-" * self.WIDTH_TOTAL + _clear_line_right())
        self._w(_goto(self.ROW_L1, self.LEFT_COL) +
                _c("1", "EXEC SPREADS (instant)", self.opts.color) + "  " + inst_line[:self.LEFT_WIDTH] + _clear_line_right())
        self._w(_goto(self.ROW_L2, self.LEFT_COL) +
                _c("1", "EXEC SPREADS (5m median)", self.opts.color) + "  " + med_line[:self.LEFT_WIDTH] + _clear_line_right())

        fair = state.fair.compute(book)
        fair_txt = _fmt_price(fair) if fair is not None else "—"
        fp_mode = state.fair.mode
        self._w(_goto(self.ROW_L3, self.LEFT_COL) +
                f"Fair price [{fp_mode}]: {fair_txt}"[:self.LEFT_WIDTH] + _clear_line_right())

        # ---------- Right panel ----------
        width = self.WIDTH_TOTAL - self.RIGHT_COL - 1
        if self.opts.show_events:
            # title + columns
            title = _c("1", "EVENTS", self.opts.color)
            hdr = (
                f"{'tsZ':<{self.RIGHT_TS_W}} "
                f"{'type':<{self.RIGHT_TYPE_W}} "
                f"{'side':<{self.RIGHT_SIDE_W}} "
                f"{'qty':>{self.RIGHT_QTY_W}} "
                f"{'price':>{self.RIGHT_PX_W}}  msg"
            )
            self._w(_goto(self.ROW_TITLES, self.RIGHT_COL) + title + _clear_line_right())
            self._w(_goto(self.ROW_RIGHT_TITLE, self.RIGHT_COL) + hdr + _clear_line_right())

            # --- FAIR PRICES en haut du panneau events (configurable) ---
            # Réserve quelques lignes pour afficher mid/micro/ewma avec les mêmes colonnes.
            fair_lines = 0
            modes = tuple(m.lower() for m in self.opts.events_fp_modes if m)
            if modes:
                for i, m in enumerate(modes):
                    # Prépare/maintient l'instance si ewma
                    if m == "ewma":
                        inst = state.fair_events.get("ewma")
                        if inst is None:
                            inst = FairPrice(mode="ewma_mid", lambda_=self.opts.fp_lambda)
                            state.fair_events["ewma"] = inst
                        val = inst.compute(book)
                        msg = f"λ={self.opts.fp_lambda:g}"
                    elif m in ("micro", "microprice"):
                        inst = FairPrice(mode="microprice", alpha=self.opts.fp_alpha)
                        val = inst.compute(book)
                        msg = f"α={self.opts.fp_alpha:g}"
                        m = "micro"
                    elif m == "mid":
                        inst = FairPrice(mode="mid")
                        val = inst.compute(book)
                        msg = ""
                    else:
                        val = None
                        msg = ""

                    typ = _c("36", f"fair:{m}", self.opts.color)
                    price_txt = _fmt_price(val) if val is not None else ""
                    # ligne alignée avec le tableau events
                    line = (
                        f"{'':<{self.RIGHT_TS_W}} "
                        f"{typ:<{self.RIGHT_TYPE_W}} "
                        f"{'':<{self.RIGHT_SIDE_W}} "
                        f"{'':>{self.RIGHT_QTY_W}} "
                        f"{price_txt:>{self.RIGHT_PX_W}}  {msg}"
                    )
                    self._w(_goto(self.ROW_RIGHT_START + fair_lines, self.RIGHT_COL) + line[:width] + _clear_line_right())
                    fair_lines += 1

            # auto-append last trade as an event (dedup by trade_id)
            last_trs: List[TradeRecord] = tape.last(1)
            if last_trs:
                tr = last_trs[-1]
                if tr.trade_id != self._last_trade_event_id:
                    self.add_event(f"trade {tr.side} { _fmt_qty(tr.qty_btc) } @ {_fmt_price(tr.price_usd_per_btc)}")
                    self._last_trade_event_id = tr.trade_id

            # render events table (newest at bottom)
            lines: List[_EventRow] = list(self.events)
            nrows_total = self.ROW_RIGHT_END - self.ROW_RIGHT_START + 1
            nrows_events = max(0, nrows_total - fair_lines)
            if len(lines) < nrows_events:
                lines = [_EventRow("", "", "")] * (nrows_events - len(lines)) + lines
            else:
                lines = lines[-nrows_events:]

            MSG_W = max(
                0,
                width
                - self.RIGHT_TS_W - 1
                - self.RIGHT_TYPE_W - 1
                - self.RIGHT_SIDE_W - 1
                - self.RIGHT_QTY_W - 1
                - self.RIGHT_PX_W - 2,
            )

            for idx in range(nrows_events):
                row = self.ROW_RIGHT_START + fair_lines + idx
                ev = lines[idx]
                color_code = {"trade": "35", "snapshot": "36", "resync": "33", "info": "37"}.get(ev.etype, "37")
                typ = _c(color_code, ev.etype or "", self.opts.color)

                if ev.side == "buy":
                    side = _c("32", "buy", self.opts.color)
                elif ev.side == "sell":
                    side = _c("31", "sell", self.opts.color)
                else:
                    side = ""

                qty = _fmt_qty(ev.qty) if ev.qty is not None else ""
                px = _fmt_price(ev.price) if ev.price is not None else ""
                msg = (ev.msg or "")[:MSG_W]

                line = (
                    f"{(ev.ts or ''):<{self.RIGHT_TS_W}} "
                    f"{typ:<{self.RIGHT_TYPE_W}} "
                    f"{side:<{self.RIGHT_SIDE_W}} "
                    f"{qty:>{self.RIGHT_QTY_W}} "
                    f"{px:>{self.RIGHT_PX_W}}  "
                    f"{msg}"
                )
                self._w(_goto(row, self.RIGHT_COL) + line + _clear_line_right())
        else:
            # title + columns for last trades
            title = _c("1", "LAST TRADES", self.opts.color)
            hdr = f"{'tsZ':<24}  {'side':<5} {'qty':>10} {'price':>14}"
            self._w(_goto(self.ROW_TITLES, self.RIGHT_COL) + title + _clear_line_right())
            self._w(_goto(self.ROW_RIGHT_TITLE, self.RIGHT_COL) + hdr + _clear_line_right())

            nrows = self.ROW_RIGHT_END - self.ROW_RIGHT_START + 1
            lastn: List[TradeRecord] = tape.last(nrows)
            lastn = lastn[-nrows:]
            pad = nrows - len(lastn)

            for i in range(pad):
                row = self.ROW_RIGHT_START + i
                self._w(_goto(row, self.RIGHT_COL) + _clear_line_right())

            import datetime as _dt
            for i, tr in enumerate(lastn):
                row = self.ROW_RIGHT_START + pad + i
                ts = _dt.datetime.fromtimestamp(tr.ts_epoch, tz=_dt.timezone.utc).isoformat().replace("+00:00", "Z")
                qty = _fmt_qty(tr.qty_btc)
                px = _fmt_price(tr.price_usd_per_btc)
                side_col = _c("32", "buy ", self.opts.color) if tr.side == "buy" else _c("31", "sell", self.opts.color)
                line = f"{ts:<24}  {side_col:<5} {qty:>10} {px:>14}"
                self._w(_goto(row, self.RIGHT_COL) + line[:width] + _clear_line_right())

        try:
            self.out.flush()
        except Exception:
            pass


# Singleton renderer
_RENDERER: Optional[LiveRenderer] = None


def render(state: LiveState, *, out: Optional[TextIO] = None, opts: Optional[RenderOptions] = None) -> None:
    """
    Render/update a two-panel TTY view.
    - First call draws static layout and hides cursor.
    - Subsequent calls only update dynamic lines.
    """
    global _RENDERER
    if _RENDERER is None or (out is not None and _RENDERER.out is not out):
        _RENDERER = LiveRenderer(out=out, opts=opts)
        _RENDERER.start(state)
    else:
        if opts is not None:
            _RENDERER.opts = opts
    _RENDERER.update(state)


def stop_render() -> None:
    """Restore cursor visibility (call on clean exit)."""
    global _RENDERER
    if _RENDERER is not None:
        _RENDERER.stop()


def log_event(msg: str) -> None:
    """Append a new event to the right panel (if events mode)."""
    global _RENDERER
    if _RENDERER is not None:
        _RENDERER.add_event(msg)
