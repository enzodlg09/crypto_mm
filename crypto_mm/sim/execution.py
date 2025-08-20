from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from statistics import median
from typing import Iterable, List, Optional, Tuple, Protocol


logger = logging.getLogger("sim.exec")


class _TradeLike(Protocol):
    """
    Contrat minimal attendu pour un trade de la bande (duck-typing).
    Compatible avec crypto_mm.core.types.TradeRecord.
    """
    ts_epoch: float
    symbol: str
    side: str                # 'buy' | 'sell'
    price_usd_per_btc: float
    qty_btc: float
    trade_id: int


@dataclass(frozen=True)
class QuotePair:
    """
    Paire de quotes symétriques pour simulation.

    Parameters
    ----------
    bid_px : float
        Prix bid (USD/BTC).
    ask_px : float
        Prix ask (USD/BTC).
    bid_qty : float
        Quantité disponible côté bid (BTC).
    ask_qty : float
        Quantité disponible côté ask (BTC).
    ts_place_epoch : float
        Timestamp UNIX (secondes, UTC) au moment de la mise en marché.
    """
    bid_px: float
    ask_px: float
    bid_qty: float
    ask_qty: float
    ts_place_epoch: float


@dataclass(frozen=True)
class FillEvent:
    """
    Événement de remplissage simulé.

    Parameters
    ----------
    side : {'buy','sell'}
        Côté de notre exécution (buy = nous achetons au bid, sell = nous vendons à l'ask).
    px : float
        Prix exécuté (USD/BTC).
    qty : float
        Quantité exécutée (BTC).
    ts_trade : float
        Timestamp UNIX de la transaction de bande utilisée.
    ts_fill : float
        Timestamp UNIX du fill effectif (ici = ts_trade).
    """
    side: str
    px: float
    qty: float
    ts_trade: float
    ts_fill: float


def _rng_bool(p: float, rng: Optional[random.Random]) -> bool:
    if p >= 1.0:
        return True
    if p <= 0.0:
        return False
    r = (rng or random).random()
    return r < p


def simulate_fills(
    trades: Iterable[_TradeLike],
    quotes: QuotePair,
    *,
    latency_ms: int = 0,
    p_priority_bid: float = 1.0,
    p_priority_ask: float = 1.0,
    rng: Optional[random.Random] = None,
) -> List[FillEvent]:
    """
    Simule les fills obtenus par une paire de quotes face à une bande de trades.

    Logique
    -------
    - Activation des quotes après `ts_place + latency_ms/1000`.
    - Ask: si trade.price >= ask_px et trade.side == 'buy' -> on peut être "hit".
    - Bid: si trade.price <= bid_px et trade.side == 'sell' -> on peut "lift".
    - Probabilité de service par côté: p_priority_{bid,ask}.
    - Partiel: qty = min(trade.qty, remaining_quote_qty).

    Parameters
    ----------
    trades : Iterable[_TradeLike]
        Bande de transactions **triée** par temps croissant (ts_epoch).
    quotes : QuotePair
        Paire de quotes à simuler.
    latency_ms : int, optional
        Latence de mise en marché des quotes (ms).
    p_priority_bid : float, optional
        Proba d'être servi lorsqu'un trade touche le bid (0..1).
    p_priority_ask : float, optional
        Proba d'être servi lorsqu'un trade touche l'ask (0..1).
    rng : random.Random, optional
        Générateur pseudo-aléatoire (tests déterministes).

    Returns
    -------
    list[FillEvent]
        Liste ordonnée de fills simulés.
    """
    t_active = quotes.ts_place_epoch + latency_ms / 1000.0
    rem_bid = max(0.0, float(quotes.bid_qty))
    rem_ask = max(0.0, float(quotes.ask_qty))

    fills: List[FillEvent] = []

    for tr in trades:
        ts = tr.ts_epoch
        if ts < t_active:
            continue

        # Ask side — on est "hit" par un buy agressif à prix >= notre ask
        if rem_ask > 0.0 and tr.price_usd_per_btc >= quotes.ask_px and tr.side == "buy":
            if _rng_bool(p_priority_ask, rng):
                qty = min(rem_ask, tr.qty_btc)
                if qty > 0.0:
                    fills.append(FillEvent(side="sell", px=quotes.ask_px, qty=qty, ts_trade=ts, ts_fill=ts))
                    rem_ask -= qty

        # Bid side — on "lift" via un sell agressif à prix <= notre bid
        if rem_bid > 0.0 and tr.price_usd_per_btc <= quotes.bid_px and tr.side == "sell":
            if _rng_bool(p_priority_bid, rng):
                qty = min(rem_bid, tr.qty_btc)
                if qty > 0.0:
                    fills.append(FillEvent(side="buy", px=quotes.bid_px, qty=qty, ts_trade=ts, ts_fill=ts))
                    rem_bid -= qty

        if rem_bid <= 1e-12 and rem_ask <= 1e-12:
            break

    # Logging structuré (p50/p95 time-to-fill depuis placement)
    if fills:
        ttf_ms = [(f.ts_fill - quotes.ts_place_epoch) * 1000.0 for f in fills]
        ttf_ms.sort()
        p50 = ttf_ms[len(ttf_ms) // 2] if len(ttf_ms) % 2 == 1 else median(ttf_ms)
        p95 = ttf_ms[max(0, math.floor(0.95 * (len(ttf_ms) - 1)))]
        logger.info(
            {
                "event": "sim.exec.stats",
                "fills": len(fills),
                "qty_buy": round(sum(f.qty for f in fills if f.side == "buy"), 10),
                "qty_sell": round(sum(f.qty for f in fills if f.side == "sell"), 10),
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
            }
        )
    else:
        logger.info({"event": "sim.exec.stats", "fills": 0})

    return fills


def remaining_after_fills(quotes: QuotePair, fills: Iterable[FillEvent]) -> Tuple[float, float]:
    """
    Quantités quotes restantes après la série de fills.

    Returns
    -------
    (float, float)
        (remaining_bid_qty, remaining_ask_qty)
    """
    rb = quotes.bid_qty
    ra = quotes.ask_qty
    for f in fills:
        if f.side == "buy":
            rb -= f.qty
        elif f.side == "sell":
            ra -= f.qty
    return max(0.0, rb), max(0.0, ra)
