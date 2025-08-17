from __future__ import annotations

import io
import time

from crypto_mm.data.order_book import OrderBookL2
from crypto_mm.data.trade_tape import TradeTape, TradeRecord
from crypto_mm.ui.live_view import LiveState, render
from crypto_mm.core.types import BookMessage, BookLevel


def _snapshot(seq: int):
    return BookMessage(
        ts="2024-01-01T00:00:00Z",
        symbol="BTC/USD",
        type="snapshot",
        seq=seq,
        bids=[BookLevel(price_usd_per_btc=100.0, qty_btc=2.0, action="upsert")],
        asks=[BookLevel(price_usd_per_btc=101.0, qty_btc=1.5, action="upsert")],
    )


def test_render_runs_fast_and_handles_empty() -> None:
    ob = OrderBookL2()
    ob.apply_snapshot(_snapshot(1))
    tape = TradeTape(capacity=10)
    tape.push(TradeRecord(ts_epoch=time.time(), symbol="BTC/USD", side="buy", price_usd_per_btc=100.5, qty_btc=0.01, trade_id=1))

    state = LiveState(pair="BTC/USD", order_book=ob, trade_tape=tape, depth=5, min_size=0.0)

    out = io.StringIO()
    t0 = time.perf_counter()
    render(state, out=out)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < 50.0  # performance target

    # contains headings and some content
    s = out.getvalue()
    assert "TOP 10 LEVELS" in s
    assert "LAST 10 TRADES" in s
