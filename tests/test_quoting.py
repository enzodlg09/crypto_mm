from __future__ import annotations

import math

import pytest

from crypto_mm.mm.quoting import generate_quote


def test_inventory_skew_long_increases_bid_and_decreases_ask() -> None:
    q = generate_quote(
        fair_price=100.0,
        spread_quote=0.20,   # half = 0.10
        quote_size=0.5,
        tick=0.01,           # tick fin pour voir le skew
        inventory=0.50,      # long
        k_skew=0.05,         # Δ = 0.025
        min_quote_size=0.1,
    )
    # Sans skew: bid=99.90, ask=100.10
    # Avec skew Δ=+0.025: bid_pre=99.925 -> 99.93 ; ask_pre=100.075 -> 100.08
    assert math.isclose(q.bid_px, 99.93, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(q.ask_px, 100.08, rel_tol=0, abs_tol=1e-9)
    assert q.ask_px > q.bid_px
    assert q.bid_sz >= 0.1 and q.ask_sz >= 0.1


def test_inventory_skew_short_increases_ask_and_decreases_bid() -> None:
    q = generate_quote(
        fair_price=200.0,
        spread_quote=0.50,   # half = 0.25
        quote_size=1.0,
        tick=0.05,
        inventory=-2.0,      # short
        k_skew=0.05,         # Δ = -0.10
        min_quote_size=0.0,
    )
    # Sans skew:  bid=199.75, ask=200.25
    # Δ=-0.10 -> bid_pre=199.65 -> 199.65, ask_pre=200.35 -> 200.35
    assert math.isclose(q.bid_px, 199.65, abs_tol=1e-12)
    assert math.isclose(q.ask_px, 200.35, abs_tol=1e-12)
    assert q.ask_px > q.bid_px


def test_tick_bigger_than_half_spread_widens_to_avoid_cross() -> None:
    q = generate_quote(
        fair_price=100.0,
        spread_quote=0.20,  # half = 0.10
        quote_size=0.1,
        tick=1.00,          # tick >> half-spread
        inventory=0.49,     # Δ=0.49 -> bid_pre=100.39 floor=100 ; ask_pre=99.61 ceil=100 -> clash
        k_skew=1.0,
    )
    # Doit forcer ask >= bid + tick (ici 101 vs 100 après snapping)
    assert q.ask_px - q.bid_px >= 1.00 - 1e-12
    assert q.ask_px > q.bid_px


def test_bucket_size_used_when_no_tick() -> None:
    q = generate_quote(
        fair_price=50.0,
        spread_quote=1.0,
        quote_size=0.2,
        bucket_size=0.5,    # pas de tick fourni -> bucket utilisé
        inventory=0.0,
        k_skew=0.0,
    )
    # Sans skew: bid=49.5 -> 49.5 ; ask=50.5 -> 50.5
    assert math.isclose(q.bid_px, 49.5, abs_tol=1e-12)
    assert math.isclose(q.ask_px, 50.5, abs_tol=1e-12)


def test_min_size_clamp() -> None:
    q = generate_quote(
        fair_price=10.0,
        spread_quote=0.2,
        quote_size=0.01,     # plus petit que min
        tick=0.01,
        inventory=0.0,
        k_skew=0.0,
        min_quote_size=0.05,
    )
    assert q.bid_sz == 0.05
    assert q.ask_sz == 0.05
