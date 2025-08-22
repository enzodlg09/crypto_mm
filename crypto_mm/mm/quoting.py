from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from decimal import Decimal, ROUND_HALF_UP, ROUND_CEILING

@dataclass(frozen=True)
class Quote:
    """Quote bid/ask finalisée (prix en USD/BTC, tailles en BTC)."""
    bid_px: float
    ask_px: float
    bid_sz: float
    ask_sz: float
    center_raw: float
    center_skewed: float


# ---------- precise step rounding helpers ----------

def _step_from(tick: Optional[float], bucket_size: Optional[float]) -> Decimal:
    if tick is not None and tick > 0:
        return Decimal(str(tick))
    if bucket_size is not None and bucket_size > 0:
        return Decimal(str(bucket_size))
    return Decimal("0.01")


def _round_to_step_half_up(px: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return px
    units = (px / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    out = (units * step)
    return out.quantize(step)


def _ceil_to_step(x: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return x
    units = (x / step).to_integral_value(rounding=ROUND_CEILING)
    return (units * step).quantize(step)


def _ensure_min_separation_decimal(bid: Decimal, ask: Decimal, step: Decimal) -> tuple[Decimal, Decimal]:
    min_ask = bid + step
    if ask < min_ask:
        ask = _ceil_to_step(min_ask, step)
    return bid, ask


# ---------- main ----------

def generate_quote(
    *,
    fair_price: float,
    spread_quote: float,
    quote_size: float,
    inventory: float,
    k_skew: float,
    tick: Optional[float] = None,
    bucket_size: Optional[float] = None,
    min_quote_size: float = 0.0,
) -> Quote:
    """
    Calcule une cotation bid/ask:
      - skew inventaire: Δ = k_skew * inventory
        bid_pre = fair - half + Δ
        ask_pre = fair + half - Δ
      - snapping: nearest HALF_UP au multiple de (tick ou bucket_size)
      - séparation forcée: ask >= bid + step
      - tailles: clamp à min_quote_size
    """
    # --- tout en Decimal pour éviter les biais binaires avant l'arrondi
    fair_d = Decimal(str(fair_price))
    half_d = Decimal(str(spread_quote)) / Decimal("2")
    delta_d = Decimal(str(k_skew)) * Decimal(str(inventory))

    bid_pre_d = fair_d - half_d + delta_d
    ask_pre_d = fair_d + half_d - delta_d

    center_skewed_d = (bid_pre_d + ask_pre_d) / Decimal("2")

    step = _step_from(tick, bucket_size)

    bid_d = _round_to_step_half_up(bid_pre_d, step)
    ask_d = _round_to_step_half_up(ask_pre_d, step)

    bid_d, ask_d = _ensure_min_separation_decimal(bid_d, ask_d, step)

    qsz = float(quote_size)
    msz = float(min_quote_size or 0.0)
    final_sz = qsz if qsz >= msz else msz

    return Quote(
        bid_px=float(bid_d),
        ask_px=float(ask_d),
        bid_sz=float(final_sz),
        ask_sz=float(final_sz),
        center_raw=float(fair_d),
        center_skewed=float(center_skewed_d),
    )
