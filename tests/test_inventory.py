from __future__ import annotations

from crypto_mm.mm.inventory import Inventory
from crypto_mm.sim.execution import FillEvent


def _sell(px: float, qty: float) -> FillEvent:
    return FillEvent(side="sell", px=px, qty=qty, ts_trade=0.0, ts_fill=0.0)


def _buy(px: float, qty: float) -> FillEvent:
    return FillEvent(side="buy", px=px, qty=qty, ts_trade=0.0, ts_fill=0.0)


def test_long_sequence_and_marks() -> None:
    inv = Inventory()

    # Buy 1 @ 100 -> long 1 @ 100
    inv.on_fill(_buy(100.0, 1.0))
    inv.mark(110.0)
    assert inv.position_btc == 1.0
    assert inv.avg_entry_px == 100.0
    assert inv.pnl_realized == 0.0
    assert inv.pnl_unrealized == 10.0
    assert inv.exposure_usd == 110.0

    # Sell 0.4 @ 120 -> realise (120-100)*0.4 = 8 ; reste long 0.6 @ 100
    inv.on_fill(_sell(120.0, 0.4))
    inv.mark(110.0)
    assert inv.position_btc == 0.6
    assert inv.avg_entry_px == 100.0
    assert round(inv.pnl_realized, 10) == 8.0
    assert inv.pnl_unrealized == (110.0 - 100.0) * 0.6 == 6.0
    assert inv.pnl_total == 14.0

    # Sell 0.6 @ 90 -> realise -6 ; flat
    inv.on_fill(_sell(90.0, 0.6))
    inv.mark(110.0)
    assert inv.position_btc == 0.0
    assert inv.avg_entry_px is None
    assert inv.pnl_unrealized == 0.0
    assert round(inv.pnl_realized, 10) == 2.0


def test_short_cover_and_flip() -> None:
    inv = Inventory()

    # Short 1 @ 200
    inv.on_fill(_sell(200.0, 1.0))
    assert inv.position_btc == -1.0
    assert inv.avg_entry_px == 200.0

    # Cover 0.4 @ 180 -> realise (200-180)*0.4 = 8 ; reste -0.6 @ 200
    inv.on_fill(_buy(180.0, 0.4))
    assert inv.position_btc == -0.6
    assert inv.avg_entry_px == 200.0
    assert round(inv.pnl_realized, 10) == 8.0

    # Buy 1 @ 210 -> cover 0.6 realise (200-210)*0.6 = -6 ; flip long 0.4 @ 210
    inv.on_fill(_buy(210.0, 1.0))
    assert inv.position_btc == 0.4
    assert inv.avg_entry_px == 210.0
    assert round(inv.pnl_realized, 10) == 2.0  # 8 - 6
