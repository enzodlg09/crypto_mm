from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, TextIO, Tuple

from ..core.types import utc_now_iso
from ..data.order_book import OrderBookL2
from ..data.trade_tape import TradeTape, TradeRecord


@dataclass
class LiveState:
    """State container for live rendering."""
    pair: str
    order_book: OrderBookL2
    trade_tape: TradeTape
    depth: int = 10
    min_size: float = 0.0


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
    # Move cursor to row;col (1-based)
    return f"\x1b[{row};{col}H"


def _clear_line_right() -> str:
    # Clear from cursor to end of line
    return "\x1b[K"


class LiveRenderer:
    """
    Single-window renderer using ANSI cursor addressing.
    Left panel: OrderBook (top10, metrics, ladder).
    Right panel: Event log (snapshots, resyncs, last trades).
    """

    # Geometry (1-based rows)
    ROW_HEADER = 1
    ROW_SEP1 = 2

    ROW_TOP_TITLE = 3
    ROW_TOP_HDR = 4
    ROW_TOP_START = 5          # 10 rows (5..14)
    ROW_TOP_END = 14

    ROW_METRICS = 15
    ROW_SEP2 = 16

    ROW_LADDER_TITLE = 17
    ROW_LADDER_BIDS = 18
    ROW_LADDER_ASKS = 19

    ROW_SEP3 = 20

    ROW_EVENTS_TITLE = 3
    ROW_EVENTS_START = 4       # ~16 rows (4..19)
    ROW_EVENTS_END = 19

    WIDTH_TOTAL = 120
    LEFT_WIDTH = 80
    RIGHT_COL = LEFT_WIDTH + 3  # start col of right panel

    def __init__(self, out: Optional[TextIO] = None) -> None:
        self.out: TextIO = out if out is not None else sys.stdout
        self.started: bool = False
        self.events: Deque[str] = deque(maxlen=(self.ROW_EVENTS_END - self.ROW_EVENTS_START + 1))

    def _w(self, s: str) -> None:
        try:
            self.out.write(s)
        except (ValueError, OSError):
            # Stream might be closed; ignore to keep loop alive.
            pass

    def start(self, state: LiveState) -> None:
        if self.started:
            return
        # Clear screen, go home, hide cursor
        self._w("\x1b[2J\x1b[H\x1b[?25l")

        # Vertical separator (right panel)
        for r in range(self.ROW_TOP_TITLE, self.ROW_SEP3 + 1):
            self._w(_goto(r, self.RIGHT_COL - 2) + "│")

        # Titles / separators
        self._w(_goto(self.ROW_SEP1, 1) + "=" * self.WIDTH_TOTAL + _clear_line_right())
        self._w(_goto(self.ROW_TOP_TITLE, 2) + "ORDER BOOK — TOP 10 (cumulative sizes)" + _clear_line_right())
        hdr = f"{'BIDS (px/usd)':>18} {'cum_btc':>10}    ||    {'ASKS (px/usd)':>18} {'cum_btc':>10}"
        self._w(_goto(self.ROW_TOP_HDR, 2) + hdr + _clear_line_right())

        for r in range(self.ROW_TOP_START, self.ROW_TOP_END + 1):
            self._w(_goto(r, 2) + _clear_line_right())

        self._w(_goto(self.ROW_METRICS, 2) + _clear_line_right())
        self._w(_goto(self.ROW_SEP2, 1) + "-" * self.WIDTH_TOTAL + _clear_line_right())
        self._w(_goto(self.ROW_LADDER_TITLE, 2) + f"LADDER (min_size={state.min_size}, depth={state.depth})" + _clear_line_right())
        self._w(_goto(self.ROW_LADDER_BIDS, 2) + _clear_line_right())
        self._w(_goto(self.ROW_LADDER_ASKS, 2) + _clear_line_right())
        self._w(_goto(self.ROW_SEP3, 1) + "-" * self.WIDTH_TOTAL + _clear_line_right())

        # Right panel title & clear area
        self._w(_goto(self.ROW_EVENTS_TITLE, self.RIGHT_COL) + "EVENTS" + _clear_line_right())
        for r in range(self.ROW_EVENTS_START, self.ROW_EVENTS_END + 1):
            self._w(_goto(r, self.RIGHT_COL) + _clear_line_right())

        self.out.flush()
        self.started = True

    def stop(self) -> None:
        # Show cursor again
        self._w("\x1b[?25h")
        try:
            self.out.flush()
        except Exception:
            pass

    # ---------- Event log API ----------
    def add_event(self, msg: str) -> None:
        ts = utc_now_iso()
        short = msg.replace("\n", " ")
        self.events.append(f"{ts}  {short}")

    # ---------- Dynamic updates ----------
    def update(self, state: LiveState) -> None:
        ob = state.order_book
        tape = state.trade_tape

        # Header
        now = utc_now_iso()
        header = f"{state.pair}  @ {now}"
        self._w(_goto(self.ROW_HEADER, 2) + header + _clear_line_right())

        # Top-10 with cumulative sizes
        top = ob.top(50)
        bids = top["bids"]
        asks = top["asks"]
        cbids = _format_levels_cumsum(bids, 10)
        casks = _format_levels_cumsum(asks, 10)

        for i in range(10):
            row = self.ROW_TOP_START + i
            lb = f"{_fmt_price(cbids[i][0])} {_fmt_qty(cbids[i][1])}" if i < len(cbids) else ""
            la = f"{_fmt_price(casks[i][0])} {_fmt_qty(casks[i][1])}" if i < len(casks) else ""
            line = f"{lb:>30}    ||    {la:<30}"
            self._w(_goto(row, 2) + line + _clear_line_right())

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
        self._w(_goto(self.ROW_METRICS, 2) + metrics + _clear_line_right())

        # Ladder filtered
        ladder = ob.ladder(depth=state.depth, min_size=state.min_size)
        ladd_b = " ".join([f"{int(p)}@{_fmt_qty(q)}" for p, q in ladder["bids"][:state.depth]]) or "—"
        ladd_a = " ".join([f"{int(p)}@{_fmt_qty(q)}" for p, q in ladder["asks"][:state.depth]]) or "—"
        self._w(_goto(self.ROW_LADDER_TITLE, 2) + f"LADDER (min_size={state.min_size}, depth={state.depth})" + _clear_line_right())
        self._w(_goto(self.ROW_LADDER_BIDS, 2) + f"BIDS: {ladd_b}" + _clear_line_right())
        self._w(_goto(self.ROW_LADDER_ASKS, 2) + f"ASKS: {ladd_a}" + _clear_line_right())

        # Right panel: events (most recent at bottom)
        lines: List[str] = list(self.events)
        nrows = self.ROW_EVENTS_END - self.ROW_EVENTS_START + 1
        if len(lines) < nrows:
            lines = [""] * (nrows - len(lines)) + lines  # pad top with blanks
        else:
            lines = lines[-nrows:]

        for idx in range(nrows):
            row = self.ROW_EVENTS_START + idx
            self._w(_goto(row, self.RIGHT_COL) + lines[idx][: (self.WIDTH_TOTAL - self.RIGHT_COL - 1)] + _clear_line_right())

        # (Optional) echo newest trades as events (lightweight)
        # Keep only very recent trades in the event log (the tape itself is kept elsewhere)
        last_trs: List[TradeRecord] = tape.last(1)
        if last_trs:
            tr = last_trs[-1]
            side = tr.side
            qty = _fmt_qty(tr.qty_btc)
            px = _fmt_price(tr.price_usd_per_btc)
            # Only append if not the same as last line to reduce spam
            if not self.events or "trade" not in self.events[-1]:
                self.add_event(f"trade {side} {qty} @ {px}")

        try:
            self.out.flush()
        except Exception:
            pass


# Singleton renderer used by helper functions
_RENDERER: Optional[LiveRenderer] = None


def render(state: LiveState, *, out: Optional[TextIO] = None) -> None:
    """
    Render/update a single-window TTY view.
    - First call draws static layout and hides cursor.
    - Subsequent calls only update dynamic lines.
    """
    global _RENDERER
    if _RENDERER is None or (out is not None and _RENDERER.out is not out):
        _RENDERER = LiveRenderer(out=out)
        _RENDERER.start(state)
    _RENDERER.update(state)


def stop_render() -> None:
    """Restore cursor visibility (call on clean exit)."""
    global _RENDERER
    if _RENDERER is not None:
        _RENDERER.stop()


def log_event(msg: str) -> None:
    """Append a new event to the right panel."""
    global _RENDERER
    if _RENDERER is not None:
        _RENDERER.add_event(msg)
