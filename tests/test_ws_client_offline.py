from __future__ import annotations

from typing import List

import pytest

from crypto_mm.data.ws_client import (
    _compute_checksum_v2,
    _BookState,
    _levels_from_v2,
    KrakenWSV2Client,
    ResyncRequired,
)
from crypto_mm.core.types import BookMessage, TradeMessage


# ---- Unit: converters & checksum ----

def test_levels_mapping_delete_on_zero() -> None:
    raw = [{"price": "30000.0", "qty": "0"}, {"price": "29999.5", "qty": "0.123"}]
    levels = _levels_from_v2(raw)
    assert levels[0].action == "delete" and levels[0].qty_btc == 0.0
    assert levels[1].action == "upsert" and abs(levels[1].qty_btc - 0.123) < 1e-9


def test_checksum_computation_top10() -> None:
    asks = [("50000.0", "1.0"), ("50010.0", "2.0")]
    bids = [("49990.0", "1.5"), ("49980.0", "3.0")]
    c = _compute_checksum_v2(asks, bids)
    assert c.isdigit()


# ---- Offline: order book mapping + checksum v2 ----

def _snapshot_msg(asks, bids, checksum) -> dict:
    return {
        "channel": "book",
        "type": "snapshot",
        "data": [{
            "symbol": "BTC/USD",
            "asks": asks, "bids": bids,
            "checksum": checksum
        }]
    }

def _update_msg(asks, bids, checksum) -> dict:
    return {
        "channel": "book",
        "type": "update",
        "data": [{
            "symbol": "BTC/USD",
            "asks": asks, "bids": bids,
            "checksum": checksum,
            "timestamp": "2023-10-06T17:35:55.440295Z"
        }]
    }

@pytest.mark.asyncio
async def test_order_book_sequence_and_checksum_ok(monkeypatch) -> None:
    asks = [{"price": "50000.0", "qty": "1.0"}, {"price": "50010.0", "qty": "2.0"}]
    bids = [{"price": "49990.0", "qty": "1.5"}, {"price": "49980.0", "qty": "3.0"}]
    st = _BookState.from_snapshot(asks, bids, depth=10)
    up_a = [{"price": "50005.0", "qty": "0.5"}]
    up_b = [{"price": "49995.0", "qty": "0.1"}]
    st.apply_update(up_a, up_b)
    checksum = _compute_checksum_v2(st.top10_asks(), st.top10_bids())

    messages = [
        _snapshot_msg(asks, bids, checksum),
        _update_msg(up_a, up_b, checksum),
    ]

    async def fake_stream(*, channel, symbols, **kwargs):
        for m in messages:
            yield m

    monkeypatch.setattr(KrakenWSV2Client, "_resilient_stream", lambda self, **kw: fake_stream(**kw))

    client = KrakenWSV2Client()
    got: List[BookMessage] = []
    async for m in client.subscribe_order_book("BTC/USD", depth=10):
        got.append(m)
        if len(got) >= 2:
            break

    assert got[0].type == "snapshot" and got[0].seq == 1
    assert got[1].type == "delta" and got[1].seq == 2


@pytest.mark.asyncio
async def test_order_book_checksum_mismatch_raises(monkeypatch) -> None:
    asks = [{"price": "50000.0", "qty": "1.0"}]
    bids = [{"price": "49990.0", "qty": "1.5"}]
    up_a = [{"price": "50005.0", "qty": "0.5"}]
    up_b = [{"price": "49995.0", "qty": "0.1"}]

    messages = [
        _snapshot_msg(asks, bids, "1"),
        _update_msg(up_a, up_b, "0"),  # wrong checksum
    ]

    async def fake_stream(*, channel, symbols, **kwargs):
        for m in messages:
            yield m

    monkeypatch.setattr(KrakenWSV2Client, "_resilient_stream", lambda self, **kw: fake_stream(**kw))

    client = KrakenWSV2Client()
    with pytest.raises(ResyncRequired):
        async for _ in client.subscribe_order_book("BTC/USD", depth=10):
            pass


# ---- Offline: trades mapping v2 ----

@pytest.mark.asyncio
async def test_trade_mapping_offline(monkeypatch) -> None:
    trade1 = {"symbol": "BTC/USD", "side": "buy", "price": 30000.1, "qty": 0.01, "ord_type": "limit", "trade_id": 10, "timestamp": "2023-09-25T07:48:36.925533Z"}
    trade2 = {"symbol": "BTC/USD", "side": "sell", "price": 30001.2, "qty": 0.02, "ord_type": "market", "trade_id": 11, "timestamp": "2023-09-25T07:49:36.925603Z"}

    async def fake_stream(*, channel, symbols, **kwargs):
        yield {"channel":"trade","type":"snapshot","data":[trade1]}
        yield {"channel":"heartbeat"}  # ignored
        yield {"channel":"trade","type":"update","data":[trade2]}

    monkeypatch.setattr(KrakenWSV2Client, "_resilient_stream", lambda self, **kw: fake_stream(**kw))

    client = KrakenWSV2Client()
    got: List[TradeMessage] = []
    async for t in client.subscribe_trades("BTC/USD"):
        got.append(t)
        if len(got) == 2:
            break

    assert got[0].price_usd_per_btc > 0
    assert got[0].side in ("buy", "sell")
    assert got[1].side == "sell"
