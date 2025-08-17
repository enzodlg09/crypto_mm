from __future__ import annotations

from crypto_mm.core.clock import Clock
from crypto_mm.core.log import get_logger


def test_clock_runs_and_reports_metrics() -> None:
    logger = get_logger("test", "ERROR")
    clock = Clock(period_ms=100, logger=logger)

    ticks = 0
    def cb(_i: int, _dt_ms: float) -> None:
        nonlocal ticks
        ticks += 1

    clock.run(duration_s=0.5, iterations=None, on_tick=cb)
    assert ticks >= 4
    assert clock.p50_ms() >= 0.0
    assert clock.p95_ms() >= clock.p50_ms()
    event = clock.metrics_summary_event("done")
    assert "latency_p50_ms" in event and "latency_p95_ms" in event and event["samples"] >= 1
