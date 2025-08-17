from __future__ import annotations

from datetime import datetime, timezone, timedelta

from crypto_mm.data.trade_tape import TradeTape, TradeRecord


def _ts(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def test_vwap_and_volumes_with_windows() -> None:
    now = datetime(2024, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    tape = TradeTape(capacity=10)

    # older trade (t=0s)
    tape.push(TradeRecord(ts_epoch=(now - timedelta(seconds=10)).timestamp(), symbol="BTC/USD", side="buy", price_usd_per_btc=100.0, qty_btc=1.0, trade_id=1))
    # in-window trades (t=6..9)
    tape.push(TradeRecord(ts_epoch=(now - timedelta(seconds=4)).timestamp(), symbol="BTC/USD", side="sell", price_usd_per_btc=102.0, qty_btc=2.0, trade_id=2))
    tape.push(TradeRecord(ts_epoch=(now - timedelta(seconds=2)).timestamp(), symbol="BTC/USD", side="buy",  price_usd_per_btc=101.0, qty_btc=3.0, trade_id=3))

    # Window = 5s -> includes trades at t=6..10 → last two
    vwap = tape.vwap(window_s=5, now_epoch=now.timestamp())
    # VWAP = (102*2 + 101*3) / (2+3) = (204 + 303) / 5 = 507/5 = 101.4
    assert vwap is not None
    assert abs(vwap - 101.4) < 1e-9

    vol_buy = tape.volume("buy", window_s=5, now_epoch=now.timestamp())
    vol_sell = tape.volume("sell", window_s=5, now_epoch=now.timestamp())
    assert abs(vol_buy - 3.0) < 1e-9
    assert abs(vol_sell - 2.0) < 1e-9


def test_recent_and_last_and_empty_window() -> None:
    now = datetime(2024, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    tape = TradeTape(capacity=3)

    # push out-of-order timestamps (append order: 2 -> 1 -> 3)
    t1 = now - timedelta(seconds=9)
    t2 = now - timedelta(seconds=5)
    t3 = now - timedelta(seconds=1)
    tape.push(TradeRecord(ts_epoch=t2.timestamp(), symbol="BTC/USD", side="buy", price_usd_per_btc=100.0, qty_btc=1.0, trade_id=1))
    tape.push(TradeRecord(ts_epoch=t1.timestamp(), symbol="BTC/USD", side="sell", price_usd_per_btc=99.0,  qty_btc=1.0, trade_id=2))
    tape.push(TradeRecord(ts_epoch=t3.timestamp(), symbol="BTC/USD", side="buy", price_usd_per_btc=101.0, qty_btc=1.0, trade_id=3))

    last2 = tape.last(2)
    assert [tr.trade_id for tr in last2] == [2, 3]  # append order preserved

    recent = tape.recent(_ts(now - timedelta(seconds=6)))
    # should include trades at t2 (5s ago) and t3 (1s ago)
    ids = sorted(tr.trade_id for tr in recent)
    assert ids == [1, 3]

    # empty window -> vwap None, volumes zero
    assert tape.vwap(window_s=0.5, now_epoch=(now - timedelta(seconds=1.1)).timestamp()) is None
    assert tape.volume("buy", window_s=0.5, now_epoch=(now - timedelta(seconds=1.1)).timestamp()) == 0.0
