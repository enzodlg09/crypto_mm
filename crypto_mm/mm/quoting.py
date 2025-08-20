from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from decimal import Decimal, ROUND_HALF_UP


@dataclass(frozen=True)
class Quote:
    """Quote bid/ask finalisée (prix en USD/BTC, tailles en BTC)."""
    bid_px: float
    ask_px: float
    bid_sz: float
    ask_sz: float
    center_raw: float
    center_skewed: float


def _ensure_separation_dec(bid: Decimal, ask: Decimal, step: Decimal) -> tuple[Decimal, Decimal]:
    """Garantit ask > bid (sinon élargit d'au moins 1 pas)."""
    if ask > bid:
        return (bid, ask)
    return (bid - step, ask + step)


def generate_quote(
    *,
    fair_price: float,
    spread_quote: float,
    quote_size: float,
    tick: Optional[float] = None,
    bucket_size: Optional[float] = None,
    inventory: float,
    k_skew: float,
    min_quote_size: float = 0.0,
) -> Quote:
    """
    Génère une quote bid/ask (skew symétrique + arrondi half-up au pas de cotation).

    - Centre brut: center_raw = fair_price
    - Skew Δ = k_skew * inventory
      Long  (inv>0): bid_pre = (fair-half)+Δ ; ask_pre = (fair+half)-Δ
      Short (inv<0): Δ<0 -> bid_pre ↓ ; ask_pre ↑
    - Arrondi half-up au tick si fourni, sinon bucket_size
    """
    # Choix du pas de cotation
    if tick is not None and tick > 0:
        step = Decimal(str(tick))
    elif bucket_size is not None and bucket_size > 0:
        step = Decimal(str(bucket_size))
    else:
        raise ValueError("You must provide a positive `tick` or `bucket_size`.")

    # Tout en Decimal pour éviter les erreurs binaires
    fair = Decimal(str(fair_price))
    spread = Decimal(str(spread_quote))
    half = spread / Decimal("2")
    delta = Decimal(str(k_skew)) * Decimal(str(inventory))

    # Skew symétrique
    bid_pre = (fair - half) + delta
    ask_pre = (fair + half) - delta

    # Arrondi half-up au multiple de `step`
    bid_q = (bid_pre / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    ask_q = (ask_pre / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    bid_px_dec = bid_q * step
    ask_px_dec = ask_q * step

    # Séparation minimale
    bid_px_dec, ask_px_dec = _ensure_separation_dec(bid_px_dec, ask_px_dec, step)

    # Tailles min
    sz = max(float(quote_size), float(min_quote_size))

    return Quote(
        bid_px=float(bid_px_dec),
        ask_px=float(ask_px_dec),
        bid_sz=sz,
        ask_sz=sz,
        center_raw=float(fair),
        center_skewed=float(fair),  # skew symétrique: centre conceptuel inchangé
    )
