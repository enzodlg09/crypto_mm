from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal

from dataclasses import dataclass

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Return current UTC time in ISO8601 format with 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# --- Core L2/Trades normalized types ---

Action = Literal["upsert", "delete"]
BookType = Literal["snapshot", "delta"]
Side = Literal["buy", "sell"]

class L2Level(BaseModel):
    """
    Level 2 order book level.

    Parameters
    ----------
    price_usd_per_btc : float
        Price in USD/BTC (convention).
    qty_btc : float
        Quantity in BTC (convention).
    """
    price_usd_per_btc: float = Field(..., ge=0.0)
    qty_btc: float = Field(..., ge=0.0)


class L2Snapshot(BaseModel):
    """
    Full depth snapshot (simplifié).

    Parameters
    ----------
    ts : str
        UTC ISO8601 timestamp.
    instrument : str
        Symbol (e.g., 'BTC/USD').
    bids : list[L2Level]
    asks : list[L2Level]
    """
    ts: str = Field(default_factory=utc_now_iso)
    instrument: str
    bids: List[L2Level]
    asks: List[L2Level]


class Trade(BaseModel):
    """
    Time & Sales record.

    Parameters
    ----------
    ts : str
        UTC ISO8601 timestamp.
    instrument : str
    side : {'buy','sell'}
        Taker side.
    price_usd_per_btc : float
    qty_btc : float
    amount_usd : float
    """
    ts: str = Field(default_factory=utc_now_iso)
    instrument: str
    side: Literal["buy", "sell"]
    price_usd_per_btc: float = Field(..., ge=0.0)
    qty_btc: float = Field(..., ge=0.0)
    amount_usd: float = Field(..., ge=0.0)

    @classmethod
    def from_price_qty(
        cls, instrument: str, side: Literal["buy", "sell"], price_usd_per_btc: float, qty_btc: float
    ) -> "Trade":
        amount = price_usd_per_btc * qty_btc
        return cls(instrument=instrument, side=side, price_usd_per_btc=price_usd_per_btc, qty_btc=qty_btc, amount_usd=amount)


# --- Normalized messages for WS client ---

class BookLevel(BaseModel):
    """Normalized price level for WS mapping."""
    price_usd_per_btc: float = Field(..., ge=0.0)
    qty_btc: float = Field(..., ge=0.0)
    action: Action


class BookMessage(BaseModel):
    """Normalized order book message (snapshot/delta)."""
    ts: str
    symbol: str
    type: BookType
    seq: int
    bids: List[BookLevel]
    asks: List[BookLevel]


class TradeMessage(BaseModel):
    """Normalized trade message."""
    ts: str
    symbol: str
    price_usd_per_btc: float = Field(..., ge=0.0)
    qty_btc: float = Field(..., ge=0.0)
    side: Literal["buy", "sell"]
    trade_id: int


@dataclass(frozen=True)
class TradeRecord:
    """
    Trade tape record (dataclass, utilisé par tests & simulateur).

    Notes
    -----
    - `ts_epoch` : timestamp UNIX (secondes UTC).
    - `side` : côté agressif ('buy' ou 'sell').
    - Les unités suivent les conventions du projet (USD/BTC, BTC).

    Parameters
    ----------
    ts_epoch : float
    symbol : str
    side : {'buy','sell'}
    price_usd_per_btc : float
    qty_btc : float
    trade_id : int
    """
    ts_epoch: float
    symbol: str
    side: Side
    price_usd_per_btc: float
    qty_btc: float
    trade_id: int