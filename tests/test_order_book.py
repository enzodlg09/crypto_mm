from __future__ import annotations

from typing import List, Tuple

from crypto_mm.data.order_book import OrderBookL2
from crypto_mm.core.types import BookMessage, BookLevel


def _snapshot(seq: int, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> BookMessage:
    return BookMessage(
        ts="2024-01-01T00:00:00Z",
        symbol="BTC/USD",
        type="snapshot",
        seq=seq,
        bids=[BookLevel(price_usd_per_btc=p, qty_btc=q, action="upsert") for p, q in bids],
        asks=[BookLevel(price_usd_per_btc=p, qty_btc=q, action="upsert") for p, q in asks],
    )


def _delta(seq: int, bids: List[Tuple[float, float, str]], asks: List[Tuple[float, float, str]]) -> BookMessage:
    # bids/asks entries: (price, qty, action)
    return BookMessage(
        ts="2024-01-01T00:00:01Z",
        symbol="BTC/USD",
        type="delta",
        seq=seq,
        bids=[BookLevel(price_usd_per_btc=p, qty_btc=q, action=a) for p, q, a in bids],
        asks=[BookLevel(price_usd_per_btc=p, qty_btc=q, action=a) for p, q, a in asks],
    )


def test_snapshot_mid_and_microprice() -> None:
    ob = OrderBookL2()

    # bids: (price,size), asks: (price,size)
    snap = _snapshot(
        10,
        bids=[(100.0, 2.0), (99.0, 1.0)],
        asks=[(101.0, 1.5), (102.0, 2.0)],
    )
    ob.apply_snapshot(snap)

    top = ob.top(2)
    assert top["bids"][0] == (100.0, 2.0)
    assert top["asks"][0] == (101.0, 1.5)

    # mid = (100 + 101)/2 = 100.5
    assert abs(ob.mid_price() - 100.5) < 1e-9

    # micro = (ask*B + bid*A) / (A+B) = (101*2 + 100*1.5)/(1.5+2) = 352/3.5 = 100.571428...
    micro_expected = (101.0 * 2.0 + 100.0 * 1.5) / (1.5 + 2.0)
    assert abs(ob.microprice() - micro_expected) < 1e-9


def test_delta_update_and_ladder_filter() -> None:
    ob = OrderBookL2()
    ob.apply_snapshot(_snapshot(1, bids=[(100.0, 2.0), (99.0, 1.0)], asks=[(101.0, 1.5), (102.0, 2.0)]))

    # set best bid size to 1.0, delete best ask, add new ask 101.5 with size 3.0
    d = _delta(
        2,
        bids=[(100.0, 1.0, "upsert")],
        asks=[(101.0, 0.0, "delete"), (101.5, 3.0, "upsert")],
    )
    ob.apply_delta(d)

    top1 = ob.top(2)
    assert top1["bids"][0] == (100.0, 1.0)
    assert top1["asks"][0] == (101.5, 3.0)

    # Ladder with min_size=1.5 keeps ask 101.5 (3.0), drops bid 99.0 (1.0)
    ladder = ob.ladder(depth=5, min_size=1.5)
    assert (99.0, 1.0) not in ladder["bids"]
    assert (101.5, 3.0) in ladder["asks"]


def test_duplicate_price_aggregation_snapshot_and_delta() -> None:
    ob = OrderBookL2()

    # Snapshot with duplicated ask price 105.0 (1.0 + 0.5) -> 1.5
    snap = BookMessage(
        ts="2024-01-01T00:00:00Z",
        symbol="BTC/USD",
        type="snapshot",
        seq=5,
        bids=[BookLevel(price_usd_per_btc=100.0, qty_btc=1.0, action="upsert")],
        asks=[
            BookLevel(price_usd_per_btc=105.0, qty_btc=1.0, action="upsert"),
            BookLevel(price_usd_per_btc=105.0, qty_btc=0.5, action="upsert"),
        ],
    )
    ob.apply_snapshot(snap)
    assert ob.top(1)["asks"][0] == (105.0, 1.5)

    # Delta with duplicated bid price 99.0 (0.3 + 0.2) -> set level to 0.5
    d = _delta(
        6,
        bids=[(99.0, 0.3, "upsert"), (99.0, 0.2, "upsert")],
        asks=[],
    )
    ob.apply_delta(d)
    bids = dict(ob.top(3)["bids"])
    assert abs(bids.get(99.0, 0.0) - 0.5) < 1e-12


def test_out_of_order_delta_rejected_and_flag_resync() -> None:
    ob = OrderBookL2()
    ob.apply_snapshot(_snapshot(10, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)]))

    # Gap (12 != 11) -> reject + need_resync
    d_gap = _delta(12, bids=[(100.0, 2.0, "upsert")], asks=[])
    ob.apply_delta(d_gap)

    assert ob.need_resync is True
    # book unchanged
    assert ob.top(1)["bids"][0] == (100.0, 1.0)
