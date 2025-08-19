from __future__ import annotations

import math
from typing import Dict, List, Tuple

import pytest

from crypto_mm.mm.fair_price import FairPrice

Book = Dict[str, List[Tuple[float, float]]]


def book(bid_px: float, bid_q: float, ask_px: float, ask_q: float) -> Book:
    return {"bids": [(bid_px, bid_q)], "asks": [(ask_px, ask_q)]}


def test_mid() -> None:
    fp = FairPrice(mode="mid")
    b = book(100.0, 1.0, 102.0, 2.0)
    assert fp.compute(b) == 101.0


def test_microprice_alpha_edges() -> None:
    b = book(100.0, 2.0, 102.0, 4.0)
    fp0 = FairPrice(mode="microprice", alpha=0.0)
    fp1 = FairPrice(mode="microprice", alpha=1.0)
    assert fp0.compute(b) == 100.0
    assert fp1.compute(b) == 102.0

    # denom zero -> None
    b2 = book(100.0, 0.0, 102.0, 0.0)
    fp = FairPrice(mode="microprice", alpha=0.3)
    assert fp.compute(b2) is None


def test_ewma_mid_deterministic() -> None:
    fp = FairPrice(mode="ewma_mid", lambda_=2.0)
    t0 = 1_000_000.0
    b1 = book(100.0, 1.0, 102.0, 1.0)  # mid=101
    b2 = book(110.0, 1.0, 112.0, 1.0)  # mid=111

    v0 = fp.compute(b1, now_epoch=t0)
    assert v0 == 101.0

    # dt = 1s, w = exp(-lambda*dt) = exp(-2) ~ 0.135335
    v1 = fp.compute(b2, now_epoch=t0 + 1.0)
    w = math.exp(-2.0 * 1.0)
    expected = w * 101.0 + (1 - w) * 111.0
    assert abs(v1 - expected) < 1e-12

    # empty book resets
    assert fp.compute({"bids": [], "asks": []}, now_epoch=t0 + 2.0) is None
    # after reset, new init
    assert fp.compute(b1, now_epoch=t0 + 3.0) == 101.0
