from __future__ import annotations

from datetime import datetime, timezone, timedelta

from crypto_mm.sim.execution import QuotePair, simulate_fills, remaining_after_fills
from crypto_mm.core.types import TradeRecord


def _ts(dt: datetime) -> float:
    return dt.replace(tzinfo=timezone.utc).timestamp()


def _tr(ts: float, side: str, px: float, qty: float, trade_id: int) -> TradeRecord:
    # TradeRecord(ts_epoch, symbol, side, price_usd_per_btc, qty_btc, trade_id)
    return TradeRecord(
        ts_epoch=ts,
        symbol="BTC/USD",
        side=side,
        price_usd_per_btc=px,
        qty_btc=qty,
        trade_id=trade_id,
    )


def test_simple_full_fill_ask_and_bid() -> None:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    q = QuotePair(bid_px=99.9, ask_px=100.1, bid_qty=0.3, ask_qty=0.2, ts_place_epoch=_ts(now))

    trades = [
        _tr(_ts(now + timedelta(seconds=1)), "buy", 100.11, 0.15, 1),  # hit ask 0.15
        _tr(_ts(now + timedelta(seconds=2)), "buy", 100.20, 0.10, 2),  # hit ask remaining 0.05 (partiel)
        _tr(_ts(now + timedelta(seconds=3)), "sell", 99.80, 0.30, 3),  # lift bid 0.30
    ]

    fills = simulate_fills(trades, q, latency_ms=0, p_priority_bid=1.0, p_priority_ask=1.0)
    # Ordre temporel
    assert [f.side for f in fills] == ["sell", "sell", "buy"]
    # Conservation
    rb, ra = remaining_after_fills(q, fills)
    assert rb == 0.0
    assert ra == 0.0


def test_partial_over_multiple_trades_until_exhaustion() -> None:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    q = QuotePair(bid_px=50.0, ask_px=50.5, bid_qty=0.0, ask_qty=0.3, ts_place_epoch=_ts(now))

    trades = [
        _tr(_ts(now + timedelta(seconds=1)), "buy", 50.5, 0.1, 1),
        _tr(_ts(now + timedelta(seconds=2)), "buy", 50.7, 0.25, 2),  # reste 0.2 à remplir -> 0.2
        _tr(_ts(now + timedelta(seconds=3)), "buy", 50.8, 1.0, 3),   # quote déjà épuisée -> 0
    ]

    fills = simulate_fills(trades, q, latency_ms=0, p_priority_ask=1.0)
    qty = sum(f.qty for f in fills if f.side == "sell")
    assert qty == 0.3  # pas plus que la quote
    rb, ra = remaining_after_fills(q, fills)
    assert (rb, ra) == (0.0, 0.0)


def test_latency_prevents_early_fills() -> None:
    now = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    q = QuotePair(bid_px=10.0, ask_px=10.2, bid_qty=0.1, ask_qty=0.1, ts_place_epoch=_ts(now))

    # Deux trades avant activation, un après
    trades = [
        _tr(_ts(now + timedelta(milliseconds=100)), "buy", 10.25, 0.1, 1),
        _tr(_ts(now + timedelta(milliseconds=200)), "sell", 10.00, 0.1, 2),
        _tr(_ts(now + timedelta(milliseconds=600)), "buy", 10.25, 0.1, 3),
    ]

    fills = simulate_fills(trades, q, latency_ms=500, p_priority_bid=1.0, p_priority_ask=1.0)
    # seul le dernier (après 500ms) frappe
    assert len(fills) == 1 and fills[0].side == "sell"
    rb, ra = remaining_after_fills(q, fills)
    assert (rb, ra) == (0.1, 0.0)


def test_priority_probability_extremes() -> None:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    q = QuotePair(bid_px=100.0, ask_px=101.0, bid_qty=1.0, ask_qty=1.0, ts_place_epoch=_ts(now))
    trades = [
        _tr(_ts(now + timedelta(seconds=1)), "buy", 101.2, 0.5, 1),
        _tr(_ts(now + timedelta(seconds=2)), "sell", 99.5, 0.5, 2),
    ]
    # p=0 -> aucun fill
    f0 = simulate_fills(trades, q, latency_ms=0, p_priority_bid=0.0, p_priority_ask=0.0)
    assert f0 == []
    # p=1 -> tous les fills
    f1 = simulate_fills(trades, q, latency_ms=0, p_priority_bid=1.0, p_priority_ask=1.0)
    assert len(f1) == 2
    rb, ra = remaining_after_fills(q, f1)
    assert (rb, ra) == (0.5, 0.5)


def test_sequence_and_conservation() -> None:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    q = QuotePair(bid_px=20.0, ask_px=20.2, bid_qty=0.2, ask_qty=0.15, ts_place_epoch=_ts(now))
    trades = [
        _tr(_ts(now + timedelta(seconds=1)), "sell", 19.9, 0.05, 1),
        _tr(_ts(now + timedelta(seconds=2)), "sell", 20.0, 0.20, 2),  # remplit 0.15 restant sur bid après 1er
        _tr(_ts(now + timedelta(seconds=3)), "buy", 20.3, 0.10, 3),
        _tr(_ts(now + timedelta(seconds=4)), "buy", 20.3, 0.10, 4),
    ]
    fills = simulate_fills(trades, q, latency_ms=0, p_priority_bid=1.0, p_priority_ask=1.0)
    # ordre temporel strict
    assert [f.trade_id if hasattr(f, "trade_id") else 0 for f in []] == []  # placeholder to keep style; not required
    assert [f.side for f in fills] == ["buy", "buy", "sell", "sell"]
    # conservation
    rb, ra = remaining_after_fills(q, fills)
    assert rb == 0.0 and ra == 0.0
