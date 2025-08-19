from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

BookSide = Sequence[Tuple[float, float]]  # (price_usd_per_btc, qty_btc)
Book = Dict[str, BookSide]


@dataclass
class FairPrice:
    """
    Estimateur de prix "juste" configurable.

    Modes
    -----
    - 'mid'                    : (best_ask + best_bid) / 2
    - 'microprice' (alpha)     : (α*qA*A + (1-α)*qB*B) / (α*qA + (1-α)*qB)
    - 'ewma_mid' (lambda_)     : EWMA du mid, facteur d'oubli par seconde (lambda_)

    Notes
    -----
    - Retourne None si quotes manquantes.
    - Pour 'microprice', retourne None si (α*qA + (1-α)*qB) == 0.
    - Pour 'ewma_mid', reset si book vide.
    """
    mode: str
    alpha: float = 0.5      # microprice
    lambda_: float = 1.0    # ewma_mid (par seconde)
    _state: Optional[float] = None
    _last_ts: Optional[float] = None

    def reset(self) -> None:
        self._state = None
        self._last_ts = None

    @staticmethod
    def _best_bid_ask(book: Book) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        bids = book.get("bids") or ()
        asks = book.get("asks") or ()
        best_bid = bids[0] if len(bids) > 0 else None
        best_ask = asks[0] if len(asks) > 0 else None
        return best_bid, best_ask

    def compute(self, book: Book, tape=None, now_epoch: Optional[float] = None) -> Optional[float]:
        now = float(now_epoch) if now_epoch is not None else time.time()
        mode = self.mode.lower()

        best_bid, best_ask = self._best_bid_ask(book)

        if mode == "mid":
            if not best_bid or not best_ask:
                return None
            return (best_ask[0] + best_bid[0]) / 2.0

        if mode == "microprice":
            if not best_bid or not best_ask:
                return None
            bid_px, bid_q = best_bid
            ask_px, ask_q = best_ask
            a = min(max(self.alpha, 0.0), 1.0)
            denom = a * ask_q + (1.0 - a) * bid_q
            if denom <= 0.0:
                return None
            return (a * ask_q * ask_px + (1.0 - a) * bid_q * bid_px) / denom

        if mode == "ewma_mid":
            if not best_bid or not best_ask:
                # reset si book vide
                self.reset()
                return None
            mid = (best_ask[0] + best_bid[0]) / 2.0
            if self._state is None or self._last_ts is None:
                self._state = mid
                self._last_ts = now
                return mid
            dt = max(0.0, now - self._last_ts)
            w = math.exp(-self.lambda_ * dt)
            self._state = w * self._state + (1.0 - w) * mid
            self._last_ts = now
            return self._state

        raise ValueError(f"Unknown fair price mode: {self.mode}")
