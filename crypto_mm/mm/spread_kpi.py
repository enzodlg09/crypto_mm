from __future__ import annotations
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


def _vwap(levels: Sequence[Tuple[float, float]], size_btc: float) -> Optional[float]:
    """VWAP pour consommer `size_btc` sur une liste triée (bids desc, asks asc)."""
    if size_btc <= 0.0:
        return None
    rem = float(size_btc)
    cost = 0.0
    taken = 0.0
    for px, qty in levels:
        if qty <= 0.0:
            continue
        take = qty if rem > qty else rem
        cost += take * px
        taken += take
        rem -= take
        if rem <= 1e-12:
            break
    if taken + 1e-12 < size_btc:
        return None
    return cost / taken


def spread_for_size(book: Dict[str, Sequence[Tuple[float, float]]], size_btc: float) -> Optional[float]:
    """
    Spread exécutable pour `size_btc`: VWAP(asks,size) - VWAP(bids,size).
    Retourne None si profondeur insuffisante ou si size<=0.
    """
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if not bids or not asks or size_btc <= 0.0:
        return None
    bid_vwap = _vwap(bids, size_btc)
    ask_vwap = _vwap(asks, size_btc)
    if bid_vwap is None or ask_vwap is None:
        return None
    return ask_vwap - bid_vwap


class KpiStore:
    """Stocke des mesures de spread exécutable et fournit des agrégats glissants."""

    def __init__(self, window_s: int = 300) -> None:
        self.window_s = int(window_s)
        # colonnes typées pour éviter les warnings de concat/NA
        self._df = pd.DataFrame({"ts": pd.Series(dtype="float64"),
                                 "size_btc": pd.Series(dtype="float64"),
                                 "value": pd.Series(dtype="float64")})

    def append(self, ts_epoch: float, size_btc: float, value: Optional[float]) -> None:
        """
        Ajoute un point si `value` est finie (ni None, ni NaN/inf), puis trim la fenêtre.
        """
        if value is None:
            return
        try:
            v = float(value)
        except Exception:
            return
        if not math.isfinite(v):
            return

        rec = {"ts": float(ts_epoch), "size_btc": float(size_btc), "value": v}
        # Pas de concat (évite FutureWarning), on append via .loc
        self._df.loc[len(self._df)] = rec
        self._trim(ts_epoch)

    def _trim(self, now_epoch: float) -> None:
        """Conserve uniquement la fenêtre [now - window_s, now]."""
        if self._df.empty:
            return
        cutoff = float(now_epoch) - float(self.window_s)
        self._df = self._df[self._df["ts"] >= cutoff]

    def aggregates(self, now_epoch: Optional[float] = None) -> pd.DataFrame:
        """
        Retourne un DataFrame indexé par `size_btc` avec:
          - median, p25, p75, count
        """
        if self._df.empty:
            return pd.DataFrame(columns=["median", "p25", "p75", "count"])
        g = self._df.groupby("size_btc")["value"]
        out = pd.DataFrame({
            "median": g.median(),
            "p25": g.quantile(0.25),
            "p75": g.quantile(0.75),
            "count": g.count(),
        })
        return out
