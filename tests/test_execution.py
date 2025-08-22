from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from crypto_mm.sim.execution import QuotePair, FillEvent, simulate_fills, remaining_after_fills


@dataclass
class _T:
    ts_epoch: float
    symbol: str
    side: str
    price_usd_per_btc: float
    qty_btc: float
    trade_id: int


def test_simple_sequence_no_latency() -> None:
    # quotes au temps 0
    q = QuotePair(bid_px=99.0, ask_px=100.0, bid_qty=0.5, ask_qty=0.5, ts_place_epoch=0.0)

    trades: List[_T] = [
        _T(0.10, "BTC/USD", "buy", 100.20, 0.30, 1),   # touche ask -> 0.30 sell
        _T(0.15, "BTC/USD", "buy", 100.50, 0.30, 2),   # touche ask -> 0.20 sell (reste 0.10 trade)
        _T(0.20, "BTC/USD", "sell", 98.90, 0.60, 3),   # touche bid -> 0.50 buy
    ]

    fills = simulate_fills(trades, q, latency_ms=0, p_priority_bid=1.0, p_priority_ask=1.0)
    assert [ (f.side, f.px, round(f.qty, 2)) for f in fills ] == [
        ("sell", 100.0, 0.30),
        ("sell", 100.0, 0.20),
        ("buy",  99.0,  0.50),
    ]
    rb, ra = remaining_after_fills(q, fills)
    assert rb == 0.0 and ra == 0.0


def test_latency_blocks_early_trades() -> None:
    q = QuotePair(bid_px=99.0, ask_px=100.0, bid_qty=1.0, ask_qty=1.0, ts_place_epoch=0.0)
    trades = [
        _T(0.10, "BTC/USD", "buy", 100.5, 0.7, 1),   # trop tôt si latence 300ms
        _T(0.40, "BTC/USD", "buy", 100.5, 0.7, 2),   # actif -> rempli 0.7
    ]
    fills = simulate_fills(trades, q, latency_ms=300)
    assert len(fills) == 1
    assert fills[0].side == "sell" and fills[0].qty == 0.7


def test_priority_zero_yields_no_fills() -> None:
    q = QuotePair(bid_px=99.0, ask_px=100.0, bid_qty=1.0, ask_qty=1.0, ts_place_epoch=0.0)
    trades = [
        _T(1.0, "BTC/USD", "buy", 100.5, 0.5, 1),
        _T(1.1, "BTC/USD", "sell", 98.9, 0.5, 2),
    ]
    rng = random.Random(42)
    fills = simulate_fills(trades, q, p_priority_bid=0.0, p_priority_ask=0.0, rng=rng)
    assert fills == []


def test_multiple_hits_until_exhaustion() -> None:
    q = QuotePair(bid_px=99.0, ask_px=100.0, bid_qty=0.2, ask_qty=0.5, ts_place_epoch=0.0)
    trades = [
        _T(0.1, "BTC/USD", "buy", 100.1, 0.1, 1),
        _T(0.2, "BTC/USD", "buy", 100.2, 0.2, 2),
        _T(0.3, "BTC/USD", "buy", 100.3, 0.3, 3),
    ]
    fills = simulate_fills(trades, q)
    # 0.1 + 0.2 + 0.2(partiel) = 0.5
    assert round(sum(f.qty for f in fills if f.side == "sell"), 10) == 0.5
