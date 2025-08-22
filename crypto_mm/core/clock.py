from __future__ import annotations

import time
from typing import Callable, List, Optional

import numpy as np

from .log import LoggerLike
from .types import utc_now_iso


class Clock:
    """
    Simple heartbeat clock that ticks every `period_ms` and tracks loop latencies.

    Notes
    -----
    - Timestamps: UTC ISO8601
    - Latency metrics: p50/p95 of per-tick loop delays (ms)
    """

    def __init__(self, period_ms: int, logger: LoggerLike):
        self.period_ms = period_ms
        self._period_s = period_ms / 1000.0
        self._latencies_ms: List[float] = []
        self._logger = logger

    def run(
        self,
        *,
        duration_s: Optional[float],
        iterations: Optional[int],
        on_tick: Optional[Callable[[int, float], None]] = None,
    ) -> None:
        """
        Run the heartbeat loop.

        Parameters
        ----------
        duration_s : float ou None
            Joue pour ce temps
        iterations : int or None
            Joue pour un nombre fixe de tick, si none durartion_s.
        on_tick : callable
            Callback(tick_index, dt_ms) appelle chaque tick.
        """
        start = time.perf_counter()
        next_t = start + self._period_s
        tick = 0

        def _should_continue() -> bool:
            if iterations is not None:
                return tick < iterations
            if duration_s is not None:
                return (time.perf_counter() - start) < duration_s
            return tick < 10

        while _should_continue():
            before_sleep = time.perf_counter()
            now = before_sleep
            while now < next_t:
                time.sleep(max(0.0, next_t - now))
                now = time.perf_counter()
            dt_ms = (now - before_sleep) * 1000.0
            self._latencies_ms.append(dt_ms)

            if on_tick:
                try:
                    on_tick(tick, dt_ms)
                except Exception as e:  # pragma: no cover
                    self._logger.error({"event": "tick.callback_error", "ts": utc_now_iso(), "error": str(e)})

            tick += 1
            next_t += self._period_s

    def p50_ms(self) -> float:
        return float(np.percentile(self._latencies_ms, 50)) if self._latencies_ms else 0.0

    def p95_ms(self) -> float:
        return float(np.percentile(self._latencies_ms, 95)) if self._latencies_ms else 0.0

    def metrics_summary_event(self, event: str) -> dict:
        return {
            "event": event,
            "ts": utc_now_iso(),
            "heartbeat_ms": self.period_ms,
            "latency_p50_ms": round(self.p50_ms(), 2),
            "latency_p95_ms": round(self.p95_ms(), 2),
            "samples": len(self._latencies_ms),
        }
