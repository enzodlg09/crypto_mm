from __future__ import annotations

from dataclasses import dataclass

import math

from crypto_mm.risk.limits import RiskGate, Decision


@dataclass
class Q:
    bid_px: float
    ask_px: float
    bid_qty: float
    ask_qty: float


def _p(pnl_r: float, pnl_u: float) -> dict:
    return {"pnl_realized": pnl_r, "pnl_unrealized": pnl_u}


def _inv(pos: float) -> dict:
    return {"position_btc": pos}


def test_blocks_on_hard_notional_and_recovers_after_cooldown() -> None:
    g = RiskGate(max_notional_usd=1_000.0, cooldown_hb=3)

    q = Q(99.0, 101.0, 1.0, 1.0)

    # position trop grosse pour mid=2000 -> block
    d1 = g.evaluate(q, _inv(1.0), _p(0.0, 0.0), mid=2000.0)
    assert d1.action == "block" and d1.reason.startswith("hard.notional")

    # pendant cooldown -> toujours block
    assert g.evaluate(q, _inv(0.0), _p(0.0, 0.0), mid=0.0).action == "block"
    assert g.evaluate(q, _inv(0.0), _p(0.0, 0.0), mid=0.0).action == "block"

    # cooldown écoulé + notional OK -> allow
    d4 = g.evaluate(q, _inv(0.1), _p(0.0, 0.0), mid=1000.0)
    assert d4.action == "allow"


def test_blocks_on_hard_loss_with_hysteresis() -> None:
    g = RiskGate(max_loss_usd=100.0, hysteresis_usd=10.0, cooldown_hb=2)
    q = Q(99.0, 101.0, 1.0, 1.0)

    # perte trop grande -> block
    d1 = g.evaluate(q, _inv(0.0), _p(-50.0, -70.0), mid=100.0)  # pnl_total = -120
    assert d1.action == "block" and d1.reason.startswith("hard.loss")

    # encore en cooldown
    assert g.evaluate(q, _inv(0.0), _p(0.0, -95.0), mid=100.0).action == "block"

    # pas assez de recovery (seuil hystérèse = -90) -> toujours block et relance cooldown
    d3 = g.evaluate(q, _inv(0.0), _p(0.0, -95.0), mid=100.0)
    assert d3.action == "block"

    # recovery suffisant : -85 > -90 -> après cooldown suivant, on relâche
    g.evaluate(q, _inv(0.0), _p(0.0, -85.0), mid=100.0)  # set state + cooldown
    g.evaluate(q, _inv(0.0), _p(0.0, -85.0), mid=100.0)  # still block (cooldown)
    d_ok = g.evaluate(q, _inv(0.0), _p(0.0, -85.0), mid=100.0)
    assert d_ok.action == "allow"


def test_soft_inventory_adjusts_quotes() -> None:
    g = RiskGate(inventory_cap_btc=10.0, soft_center_shift_per_excess_btc=2.0, soft_min_scale=0.5)
    q = Q(100.0, 101.0, 1.0, 1.0)

    # position 12 BTC (cap=10) -> ajustement
    d = g.evaluate(q, _inv(12.0), _p(0.0, 0.0), mid=100.5)
    assert d.action == "adjust" and d.reason == "soft.inventory"
    # tailles réduites
    assert d.bid_qty <= q.bid_qty and d.ask_qty <= q.ask_qty
    # centre décalé vers le bas (position long) => ask diminue, bid diminue
    assert d.ask_px < q.ask_px and d.bid_px <= q.bid_px


def test_soft_drawdown_triggers_cooldown() -> None:
    g = RiskGate(drawdown_trigger_usd=500.0, cooldown_hb=2)
    q = Q(100.0, 101.0, 1.0, 1.0)

    # pic : 0 -> on monte à +1000
    assert g.evaluate(q, _inv(0.0), _p(1000.0, 0.0), mid=100.5).action == "allow"
    # drawdown à 400 -> ok
    assert g.evaluate(q, _inv(0.0), _p(600.0, -200.0), mid=100.5).action == "allow"  # total = 400
    # drawdown > 500 -> block + cooldown
    d = g.evaluate(q, _inv(0.0), _p(200.0, -400.0), mid=100.5)  # total = -200 ; dd = 1200
    assert d.action == "block" and d.reason.startswith("soft.drawdown")
    # cooldown en cours
    assert g.evaluate(q, _inv(0.0), _p(200.0, -400.0), mid=100.5).action == "block"
    # cooldown écoulé
    assert g.evaluate(q, _inv(0.0), _p(200.0, -400.0), mid=100.5).action in ("allow", "adjust")
