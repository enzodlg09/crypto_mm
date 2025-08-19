from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

BookSide = Sequence[Tuple[float, float]]  # (price_usd_per_btc, qty_btc)
Book = Dict[str, BookSide]  # {"bids": [(px,qty),...], "asks": [(px,qty),...]}


def executable_px(side_levels: BookSide, size_btc: float) -> Optional[float]:
    """
    Calcule le prix moyen d'exécution (VWAP) pour consommer `size_btc` sur un côté du carnet.

    Le comportement dépend de l'ordre fourni:
    - **asks** attendus triés **ascendant** (du moins cher au plus cher)
    - **bids** attendus triés **descendant** (du plus cher au moins cher)

    La fonction consomme dans **l'ordre fourni** et prend partiellement le dernier niveau si besoin.

    Parameters
    ----------
    side_levels : Sequence[Tuple[float, float]]
        Niveaux [(price_usd_per_btc, qty_btc), ...] dans l'ordre d'exécution.
    size_btc : float
        Taille (BTC) à exécuter.

    Returns
    -------
    float | None
        Prix moyen USD/BTC si profondeur suffisante, sinon None.

    Examples
    --------
    >>> executable_px([(100.0, 0.5), (101.0, 1.0)], 0.6)
    100.16666666666667
    """
    need = float(size_btc)
    if need <= 0.0:
        return 0.0

    cost = 0.0
    taken = 0.0
    for px, qty in side_levels:
        if need <= 0:
            break
        if qty <= 0:
            continue
        take = min(need, qty)
        cost += px * take
        taken += take
        need -= take

    if taken + 1e-15 < size_btc:
        return None
    return cost / taken if taken > 0 else None


def spread_for_size(book: Book, size_btc: float) -> Optional[float]:
    """
    Spread exécutable pour `size_btc`: ask_exec(size) - bid_exec(size).

    Parameters
    ----------
    book : dict
        {"bids": [(px,qty), ... desc], "asks": [(px,qty), ... asc]}
    size_btc : float
        Taille à exécuter sur chaque côté.

    Returns
    -------
    float | None
        Spread en USD (None si profondeur insuffisante d'un côté).
    """
    ask_px = executable_px(book.get("asks", ()), size_btc)
    bid_px = executable_px(book.get("bids", ()), size_btc)
    if ask_px is None or bid_px is None:
        return None
    return ask_px - bid_px


@dataclass
class KpiStore:
    """
    Stocke les spreads exécutables et calcule des agrégations **rolling** (fenêtre glissante).

    Parameters
    ----------
    window_s : int
        Fenêtre en secondes (ex: 300 pour 5 minutes).
    """
    window_s: int = 300
    _df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["ts", "size", "spread"]))

    def append(self, ts_epoch: float, size_btc: float, spread_usd: Optional[float]) -> None:
        """Ajoute un point (ignore si spread None)."""
        if spread_usd is None:
            return
        row = pd.DataFrame({"ts": [float(ts_epoch)], "size": [float(size_btc)], "spread": [float(spread_usd)]})
        self._df = pd.concat([self._df, row], ignore_index=True)

    def _windowed(self, now_epoch: Optional[float] = None) -> pd.DataFrame:
        now = float(now_epoch) if now_epoch is not None else time.time()
        cutoff = now - self.window_s
        df = self._df
        if df.empty:
            return df
        return df[df["ts"] >= cutoff]

    def aggregates(self, now_epoch: Optional[float] = None) -> pd.DataFrame:
        """
        Agrégations rolling par taille: count, mean, median, min, max.

        Returns
        -------
        pd.DataFrame
            Index = size, colonnes = ["count","mean","median","min","max"].
        """
        win = self._windowed(now_epoch)
        if win.empty:
            return pd.DataFrame(columns=["count", "mean", "median", "min", "max"])
        g = win.groupby("size")["spread"]
        out = pd.DataFrame({
            "count": g.count().astype(int),
            "mean": g.mean(),
            "median": g.median(),
            "min": g.min(),
            "max": g.max(),
        })
        return out.sort_index()

    def latest_for_sizes(self, sizes: Iterable[float]) -> Dict[float, Optional[float]]:
        """Renvoie le dernier spread (non agrégé) vu pour chaque taille (ou None)."""
        out: Dict[float, Optional[float]] = {}
        if self._df.empty:
            return {float(s): None for s in sizes}
        for s in sizes:
            sdf = self._df[self._df["size"] == float(s)]
            out[float(s)] = None if sdf.empty else float(sdf.iloc[-1]["spread"])
        return out
