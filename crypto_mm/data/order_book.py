from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np

from ..core.log import get_logger, LoggerLike
from ..core.types import BookMessage, BookLevel


def _pct(x: List[float], p: float) -> float:
    if not x:
        return 0.0
    return float(np.percentile(x, p))


@dataclass
class OrderBookL2:
    """
    Order book L2 stateful aggregator (price levels).

    Notes
    -----
    - Asks are kept ascending by price; bids descending by price.
    - Quantities are aggregated per price within a single message (duplicate price entries).
    - Deltas are applied as *absolute* sizes per level (standard for major venues):
      - `action == "delete"` or `qty == 0` → remove the level
      - `action == "upsert"` with `qty > 0` → set the level size to that value
      When multiple upserts for the same price appear in one message, their sizes are **sommées**
      et une seule valeur finale est appliquée.
    - Sequence tracking:
      - First apply must be a snapshot; otherwise `need_resync=True`.
      - A delta with `seq != last_seq + 1` sets `need_resync=True` and is **rejeté**.
    - Conventions: price in USD/BTC, qty in BTC, timestamps UTC ISO8601 en amont.
    """

    logger: LoggerLike = field(default_factory=lambda: get_logger("order_book", "INFO"))
    bids: Dict[float, float] = field(default_factory=dict)  # price -> size
    asks: Dict[float, float] = field(default_factory=dict)  # price -> size
    last_seq: Optional[int] = None
    need_resync: bool = False

    # perf / telemetry
    _lat_apply_snap_ms: List[float] = field(default_factory=list, repr=False)
    _lat_apply_delta_ms: List[float] = field(default_factory=list, repr=False)

    # ---------- internals ----------
    @staticmethod
    def _aggregate_levels(levels: List[BookLevel]) -> Dict[float, float]:
        """Aggregate duplicated price entries by summing sizes (for a single message)."""
        agg: Dict[float, float] = {}
        for lvl in levels:
            p = float(lvl.price_usd_per_btc)
            q = float(lvl.qty_btc)
            if lvl.action == "delete" or q == 0.0:
                # mark delete as zero; real deletion handled by caller
                agg[p] = agg.get(p, 0.0)  # ensure key exists
            else:
                agg[p] = agg.get(p, 0.0) + q
        return agg

    @staticmethod
    def _apply_side_snapshot(dst: Dict[float, float], levels: List[BookLevel]) -> None:
        dst.clear()
        agg = OrderBookL2._aggregate_levels(levels)
        # keep only strictly positive qty
        for p, q in agg.items():
            if q > 0.0:
                dst[p] = q

    @staticmethod
    def _apply_side_delta(dst: Dict[float, float], levels: List[BookLevel]) -> None:
        # sum duplicates first, then apply absolute set/delete
        agg = OrderBookL2._aggregate_levels(levels)
        for p, q in agg.items():
            if q <= 0.0:
                dst.pop(p, None)
            else:
                dst[p] = q

    # ---------- public API ----------
    def clear(self) -> None:
        """Reset the book."""
        self.bids.clear()
        self.asks.clear()
        self.last_seq = None
        self.need_resync = False
        self._lat_apply_snap_ms.clear()
        self._lat_apply_delta_ms.clear()

    def apply_snapshot(self, snapshot: BookMessage) -> None:
        """
        Apply a full snapshot to reset the state.

        Parameters
        ----------
        snapshot : BookMessage
            Message of type 'snapshot'.
        """
        t0 = time.perf_counter()
        if snapshot.type != "snapshot":
            self.logger.error({"event": "ob.snapshot.invalid_type"})
            self.need_resync = True
            return
        self._apply_side_snapshot(self.bids, snapshot.bids)
        self._apply_side_snapshot(self.asks, snapshot.asks)
        self.last_seq = snapshot.seq
        self.need_resync = False
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._lat_apply_snap_ms.append(dt_ms)
        self.logger.info({
            "event": "ob.snapshot.applied",
            "seq": self.last_seq,
            "bids": len(self.bids),
            "asks": len(self.asks),
            "p50_ms": round(_pct(self._lat_apply_snap_ms, 50), 2),
            "p95_ms": round(_pct(self._lat_apply_snap_ms, 95), 2),
        })

    def apply_delta(self, delta: BookMessage) -> None:
        """
        Apply an incremental delta.

        Parameters
        ----------
        delta : BookMessage
            Message of type 'delta'. Must obey monotonic seq (`last_seq+1`).
        """
        t0 = time.perf_counter()
        if delta.type != "delta":
            self.logger.error({"event": "ob.delta.invalid_type"})
            self.need_resync = True
            return
        if self.last_seq is None:
            self.logger.error({"event": "ob.delta.without_snapshot"})
            self.need_resync = True
            return
        if delta.seq != self.last_seq + 1:
            # reject and flag resync
            self.logger.error({"event": "ob.delta.seq_gap", "last_seq": self.last_seq, "got_seq": delta.seq})
            self.need_resync = True
            return

        self._apply_side_delta(self.bids, delta.bids)
        self._apply_side_delta(self.asks, delta.asks)
        self.last_seq = delta.seq
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._lat_apply_delta_ms.append(dt_ms)
        self.logger.info({
            "event": "ob.delta.applied",
            "seq": self.last_seq,
            "bids": len(self.bids),
            "asks": len(self.asks),
            "p50_ms": round(_pct(self._lat_apply_delta_ms, 50), 2),
            "p95_ms": round(_pct(self._lat_apply_delta_ms, 95), 2),
        })

    # ---------- queries ----------
    def _sorted_bids(self) -> List[Tuple[float, float]]:
        return sorted(self.bids.items(), key=lambda kv: kv[0], reverse=True)

    def _sorted_asks(self) -> List[Tuple[float, float]]:
        return sorted(self.asks.items(), key=lambda kv: kv[0])

    def top(self, n: int) -> Dict[str, List[Tuple[float, float]]]:
        """
        Return top-n levels.

        Parameters
        ----------
        n : int
            Number of levels per side.

        Returns
        -------
        dict
            {"bids":[(price,qty)...], "asks":[...]}
        """
        return {"bids": self._sorted_bids()[:n], "asks": self._sorted_asks()[:n]}

    def mid_price(self) -> float:
        """
        Mid price (simple average) of best bid/ask.

        Returns
        -------
        float
            (best_bid + best_ask)/2 if both exist, else NaN.
        """
        bids = self._sorted_bids()
        asks = self._sorted_asks()
        if not bids or not asks:
            return float("nan")
        return (bids[0][0] + asks[0][0]) / 2.0

    def microprice(self) -> float:
        """
        Microprice: weighted by top-of-book sizes.

        Notes
        -----
        micro = (ask_price * bid_size + bid_price * ask_size) / (bid_size + ask_size)

        Returns
        -------
        float
            Microprice if both sides exist with positive sizes, else NaN.
        """
        bids = self._sorted_bids()
        asks = self._sorted_asks()
        if not bids or not asks:
            return float("nan")
        pb, qb = bids[0]
        pa, qa = asks[0]
        denom = qb + qa
        if denom <= 0.0:
            return float("nan")
        return (pa * qb + pb * qa) / denom

    def ladder(self, depth: int, min_size: float) -> Dict[str, List[Tuple[float, float]]]:
        """
        Ladder view filtered by `min_size`.

        Parameters
        ----------
        depth : int
            Max number of levels per side to return.
        min_size : float
            Filter out levels with qty < min_size.

        Returns
        -------
        dict
            {"bids":[(price,qty)...], "asks":[...]} filtered and truncated.
        """
        bids = [(p, q) for p, q in self._sorted_bids() if q >= min_size][:depth]
        asks = [(p, q) for p, q in self._sorted_asks() if q >= min_size][:depth]
        return {"bids": bids, "asks": asks}

    # ---------- metrics ----------
    def metrics_event(self, event: str) -> dict:
        """Structured metrics for logs (latencies p50/p95)."""
        return {
            "event": event,
            "apply_snapshot_p50_ms": round(_pct(self._lat_apply_snap_ms, 50), 2),
            "apply_snapshot_p95_ms": round(_pct(self._lat_apply_snap_ms, 95), 2),
            "apply_delta_p50_ms": round(_pct(self._lat_apply_delta_ms, 50), 2),
            "apply_delta_p95_ms": round(_pct(self._lat_apply_delta_ms, 95), 2),
            "levels_bids": len(self.bids),
            "levels_asks": len(self.asks),
            "last_seq": self.last_seq,
            "need_resync": self.need_resync,
        }
