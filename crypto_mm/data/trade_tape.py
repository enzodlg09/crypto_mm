from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Iterable, List, Literal, Optional, Union

from ..core.log import get_logger, LoggerLike
from ..core.types import TradeMessage


def _parse_ts_iso8601(ts: str) -> float:
    """
    Parse 'YYYY-MM-DDTHH:MM:SS(.ffffff)Z' to epoch seconds (float).

    Parameters
    ----------
    ts : str
        UTC ISO8601 (with 'Z').

    Returns
    -------
    float
        POSIX seconds since epoch (UTC).
    """
    # Support both ms/us precision
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).timestamp()


@dataclass
class TradeRecord:
    ts_epoch: float
    symbol: str
    side: Literal["buy", "sell"]
    price_usd_per_btc: float
    qty_btc: float
    trade_id: int


class TradeTape:
    """
    Ring buffer for recent trades (Time & Sales).

    Notes
    -----
    - Fixed capacity; pushing beyond capacity drops the oldest.
    - Out-of-order trades (older ts) are appended at the end (no reordering),
      but all time-window queries filter on the trade's own timestamp.
    - All sizes are quantities in BTC; prices in USD/BTC.
    """

    def __init__(self, capacity: int = 1000, logger: Optional[LoggerLike] = None):
        self.capacity = int(capacity)
        self._buf: Deque[TradeRecord] = deque(maxlen=self.capacity)
        self._logger = logger or get_logger("trade_tape", "INFO")

    # ---------- Push / Access ----------
    def push(self, trade: Union[TradeMessage, TradeRecord, dict]) -> None:
        """
        Append a trade into the ring buffer.

        Parameters
        ----------
        trade : TradeMessage | TradeRecord | dict
            Trade data. If dict, must contain keys compatible with TradeMessage.
        """
        if isinstance(trade, TradeRecord):
            rec = trade
        else:
            if isinstance(trade, TradeMessage):
                ts_epoch = _parse_ts_iso8601(trade.ts)
                rec = TradeRecord(
                    ts_epoch=ts_epoch,
                    symbol=trade.symbol,
                    side=trade.side,
                    price_usd_per_btc=float(trade.price_usd_per_btc),
                    qty_btc=float(trade.qty_btc),
                    trade_id=int(trade.trade_id),
                )
            else:
                # dict-like
                ts = str(trade["ts"])
                ts_epoch = _parse_ts_iso8601(ts)
                rec = TradeRecord(
                    ts_epoch=ts_epoch,
                    symbol=str(trade["symbol"]),
                    side="buy" if trade["side"] == "buy" else "sell",
                    price_usd_per_btc=float(trade["price_usd_per_btc"]),
                    qty_btc=float(trade["qty_btc"]),
                    trade_id=int(trade.get("trade_id", 0)),
                )
        self._buf.append(rec)

    def last(self, n: int) -> List[TradeRecord]:
        """Return the last n trades (most recent pushes), newest last."""
        n = max(0, int(n))
        if n == 0:
            return []
        # deque preserves insertion order; return tail slice
        return list(self._buf)[-n:]

    def recent(self, since_ts: str) -> List[TradeRecord]:
        """
        Return all trades with timestamp >= since_ts.

        Parameters
        ----------
        since_ts : str
            UTC ISO8601.

        Returns
        -------
        list[TradeRecord]
        """
        t0 = _parse_ts_iso8601(since_ts)
        return [tr for tr in self._buf if tr.ts_epoch >= t0]

    # ---------- Stats ----------
    def _window_records(self, window_s: float, *, now_epoch: Optional[float] = None):
        if now_epoch is None:
            now_epoch = datetime.now(timezone.utc).timestamp()
        start = now_epoch - float(window_s)
        end = now_epoch
        # inclure uniquement les trades dans [start, end]
        return (tr for tr in self._buf if start <= tr.ts_epoch <= end)

    def vwap(self, window_s: float, *, now_epoch: Optional[float] = None) -> Optional[float]:
        """
        Volume-Weighted Average Price over the last window_s seconds.

        Returns
        -------
        float | None
            VWAP in USD/BTC; None if no trades in window.
        """
        total_qty = 0.0
        total_px_qty = 0.0
        for tr in self._window_records(window_s, now_epoch=now_epoch):
            total_qty += tr.qty_btc
            total_px_qty += tr.price_usd_per_btc * tr.qty_btc
        if total_qty <= 0.0:
            return None
        return total_px_qty / total_qty

    def volume(self, side: Literal["buy", "sell"], window_s: float, *, now_epoch: Optional[float] = None) -> float:
        """
        Sum quantity (BTC) for `side` over the last window_s seconds.

        Returns
        -------
        float
            Quantity in BTC.
        """
        side = "buy" if side == "buy" else "sell"
        total = 0.0
        for tr in self._window_records(window_s, now_epoch=now_epoch):
            if tr.side == side:
                total += tr.qty_btc
        return total
