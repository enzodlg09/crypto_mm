from __future__ import annotations

from pathlib import Path

from crypto_mm.sim.replay import ReplayRunner


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(__import__("json").dumps(obj) + "\n")


def test_replay_deterministic_basic(tmp_path: Path) -> None:
    """
    Cas simple 100% déterministe :
      - seed_mid = 100.0
      - Événements trades : 100.50 puis 99.50
      - spread=1.0 → half=0.5, tick=0.1, quote_size=0.5
      Cycle 1 : fair=100.0 => bid=99.5 / ask=100.5 → trade à 100.5 -> fill SELL à 100.5 (0.5)
      Cycle 2 : fair (inchangé pour la quote, puis trade) -> bid=99.5 → trade à 99.5 -> fill BUY à 99.5 (0.5)
      PnL réalisé : (100.5 - 99.5) * 0.5 = 0.50
      Position finale : flat → PnL_total == PnL_realized
      Avg cycle (2 évts, dt 1s/1s, speed=1): 1000 ms
    """
    sample = [
        {"type": "trade", "ts": 0.0, "price": 100.50, "qty": 0.50},
        {"type": "trade", "ts": 1.0, "price": 99.50, "qty": 0.50},
    ]
    p = tmp_path / "sample.jsonl"
    _write_jsonl(p, sample)

    r = ReplayRunner(
        p,
        speed=1.0,
        tick=0.1,
        quote_size=0.5,
        spread_usd=1.0,
        k_skew=0.0,
        min_quote_size=0.0,
        seed_mid=100.0,
    )
    m = r.run()
    assert m.fills == 2
    assert abs(m.pnl_realized - 0.50) < 1e-12
    assert abs(m.pnl_total - 0.50) < 1e-12
    assert abs(m.avg_cycle_ms - 1000.0) < 1e-12


def test_replay_handles_book_without_initial_snapshot_using_seed(tmp_path: Path) -> None:
    """
    Cas limite : premier événement 'book' (delta), pas de snapshot précédent → on fournit seed_mid.
    On vérifie que le runner ne lève pas d'exception et produit des métriques cohérentes.
    """
    sample = [
        {"type": "book", "ts": 10.0, "best_bid": 99.9, "best_ask": 100.1},
        {"type": "trade", "ts": 11.0, "price": 100.60, "qty": 0.25},
    ]
    p = tmp_path / "sample_book.jsonl"
    _write_jsonl(p, sample)

    r = ReplayRunner(
        p,
        speed=10.0,           # la vitesse ne change pas les fills, seulement avg_cycle_ms
        tick=0.1,
        quote_size=0.25,
        spread_usd=1.0,
        k_skew=0.0,
        min_quote_size=0.0,
        seed_mid=100.0,       # amorce le fair
    )
    m = r.run()
    # Avec seed 100.0, quote ask = 100.5, trade 100.60 >= ask → 1 fill SELL de 0.25 (ouverture short).
    # PnL réalisé reste à 0.0 (non débouclé). Non-réalisé = (mid - avg_entry) * position
    # = (100.0 - 100.5) * (-0.25) = +0.125
    assert m.fills == 1
    assert abs(m.pnl_realized - 0.0) < 1e-12
    assert abs(m.pnl_unrealized - 0.125) < 1e-12
    assert abs(m.pnl_total - 0.125) < 1e-12
    # avg_cycle_ms basé sur ts diff (1000 ms / speed 10) = 100 ms
    assert abs(m.avg_cycle_ms - 100.0) < 1e-12
