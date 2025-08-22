from __future__ import annotations

import asyncio
import contextlib
import json
import ssl
import time
import uuid
import zlib
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from websockets import connect as ws_connect
from websockets.legacy.client import WebSocketClientProtocol

from ..core.config import Settings, load_settings
from ..core.log import get_logger, LoggerLike
from ..core.types import BookLevel, BookMessage, TradeMessage, utc_now_iso


KRAKEN_WSS_V2 = "wss://ws.kraken.com/v2"  # public v2


class ResyncRequired(Exception):
    """Raised when checksum mismatch detected or stream desync suspected."""


# ---------- Backoff ----------
async def backoff_schedule(base: float = 0.5, factor: float = 2.0, max_s: float = 5.0) -> AsyncIterator[float]:
    """Async generator yielding backoff delays up to max_s (<5s)."""
    delay = base
    while True:
        yield min(delay, max_s)
        delay = min(delay * factor, max_s)


# ---------- Helpers ----------
def _ms_to_iso(ms: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ms / 1000)) + f".{ms%1000:03d}Z"


def _action_from_qty(qty: float) -> str:
    return "delete" if qty == 0.0 else "upsert"


def _levels_from_v2(side_levels: List[Dict[str, Any]]) -> List[BookLevel]:
    """
    Convert Kraken v2 levels (dicts with price & qty) to normalized BookLevel list.

    Parameters
    ----------
    side_levels : list of {"price": (float|str), "qty": (float|str)}

    Returns
    -------
    list[BookLevel]
    """
    out: List[BookLevel] = []
    for lvl in side_levels:
        # Fields might be decoded as str if using parse_float=str (for checksum precision).
        price = float(lvl["price"])
        qty = float(lvl["qty"])
        out.append(BookLevel(price_usd_per_btc=price, qty_btc=qty, action=_action_from_qty(qty)))
    return out


def _strip_dot_and_leading_zeros(s: str) -> str:
    s = s.replace(".", "")
    return s.lstrip("0") or "0"


def _compute_checksum_v2(
    asks: List[Tuple[str, str]],  # ascending by price
    bids: List[Tuple[str, str]],  # descending by price
) -> str:
    """
    Compute Kraken v2 book checksum (CRC32 unsigned) for top 10 asks (low->high), then bids (high->low).

    Parameters
    ----------
    asks : list of (price_str, qty_str)
    bids : list of (price_str, qty_str)

    Returns
    -------
    str
        Unsigned 32-bit integer as string.
    """
    parts: List[str] = []
    for p, v in asks[:10]:
        parts.append(_strip_dot_and_leading_zeros(str(p)))
        parts.append(_strip_dot_and_leading_zeros(str(v)))
    for p, v in bids[:10]:
        parts.append(_strip_dot_and_leading_zeros(str(p)))
        parts.append(_strip_dot_and_leading_zeros(str(v)))
    payload = "".join(parts).encode("ascii")
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return str(crc)


@dataclass
class _BookState:
    """Maintain local book state (strings) for checksum verification."""
    asks: Dict[str, str]  # price_str -> qty_str
    bids: Dict[str, str]  # price_str -> qty_str
    depth: int

    @staticmethod
    def from_snapshot(asks: List[Dict[str, Any]], bids: List[Dict[str, Any]], depth: int) -> "_BookState":
        st = _BookState(asks={}, bids={}, depth=depth)
        for lvl in asks:
            st.asks[str(lvl["price"])] = str(lvl["qty"])
        for lvl in bids:
            st.bids[str(lvl["price"])] = str(lvl["qty"])
        return st

    def apply_update(self, asks: List[Dict[str, Any]], bids: List[Dict[str, Any]]) -> None:
        for lvl in asks:
            p, q = str(lvl["price"]), str(lvl["qty"])
            if float(q) == 0.0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = q
        for lvl in bids:
            p, q = str(lvl["price"]), str(lvl["qty"])
            if float(q) == 0.0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = q
        # Truncate to subscribed depth (guide v2)
        asks_sorted = sorted(self.asks.items(), key=lambda kv: float(kv[0]))[: self.depth]
        bids_sorted = sorted(self.bids.items(), key=lambda kv: float(kv[0]), reverse=True)[: self.depth]
        self.asks = dict(asks_sorted)
        self.bids = dict(bids_sorted)

    def top10_asks(self) -> List[Tuple[str, str]]:
        return sorted(self.asks.items(), key=lambda kv: float(kv[0]))[:10]

    def top10_bids(self) -> List[Tuple[str, str]]:
        return sorted(self.bids.items(), key=lambda kv: float(kv[0]), reverse=True)[:10]


class KrakenWSV2Client:
    """
    Async Kraken Spot websocket client (v2 public): trades + book.

    - auto-reconnect with exponential backoff (<5s cap)
    - application-level ping (v2)
    - per-connection session id
    - normalized mapping to BookMessage / TradeMessage
    - book CRC32 checksum verification (v2); mismatch -> ResyncRequired

    Notes
    -----
    * Timestamps are UTC ISO8601.
    * seq is a local, monotonically increasing counter (v2 book/trade supply their own ids/timestamps).
    """

    def __init__(self, settings: Optional[Settings] = None, logger: Optional[LoggerLike] = None):
        self.settings = settings or load_settings()
        self.logger = logger or get_logger("ws", self.settings.log_level)
        self.session_id = str(uuid.uuid4())

    # ---- low-level ----
    async def _connect(self) -> WebSocketClientProtocol:
        ssl_ctx = ssl.create_default_context()
        ws: WebSocketClientProtocol = await ws_connect(
            self.settings.websocket_url or KRAKEN_WSS_V2,
            ssl=ssl_ctx,
            ping_interval=None,
            max_queue=4096,
        )
        self.logger.info({"event": "ws.connected", "ts": utc_now_iso(), "session": self.session_id})
        return ws

    async def _send(self, ws: WebSocketClientProtocol, payload: Dict[str, Any]) -> None:
        await ws.send(json.dumps(payload))

    async def _ping(self, ws: WebSocketClientProtocol) -> None:
        await self._send(ws, {"method": "ping", "req_id": int(time.time() * 1000)})
        self.logger.debug({"event": "ws.ping", "ts": utc_now_iso()})

    async def _subscribe(self, ws: WebSocketClientProtocol, channel: str, *, symbols: List[str], **kwargs: Any) -> None:
        params = {"channel": channel, "symbol": symbols} | kwargs
        await self._send(ws, {"method": "subscribe", "params": params})
        self.logger.info({"event": "ws.subscribe", "ts": utc_now_iso(), "channel": channel, "params": params})

    async def _recv_json(self, ws: WebSocketClientProtocol) -> Any:
        raw = await ws.recv()
        return json.loads(raw, parse_float=str)

    # ---- public APIs ----
    async def subscribe_trades(self, pair: str) -> AsyncIterator[TradeMessage]:
        """
        Async iterator over normalized trades (Kraken v2).

        Parameters
        ----------
        pair : str
            Kraken symbol, e.g., 'BTC/USD'.

        Yields
        ------
        TradeMessage
        """
        local_seq = 0
        async for payload in self._resilient_stream(channel="trade", symbols=[pair]):
            if not isinstance(payload, dict):
                continue
            if payload.get("channel") != "trade" or "data" not in payload:
                continue
            for trade in payload["data"]:
                local_seq += 1
                yield TradeMessage(
                    ts=str(trade["timestamp"]),
                    symbol=str(trade["symbol"]),
                    price_usd_per_btc=float(trade["price"]),
                    qty_btc=float(trade["qty"]),
                    side="buy" if trade["side"] == "buy" else "sell",
                    trade_id=int(trade["trade_id"]),
                )

    async def subscribe_order_book(self, pair: str, *, depth: int = 10) -> AsyncIterator[BookMessage]:
        """
        Async iterator over normalized order book messages (snapshot + deltas + checksum v2).

        Parameters
        ----------
        pair : str
            Kraken pair (e.g., 'BTC/USD').
        depth : int
            Subscription depth (10, 25, 100, 500, 1000).
        """
        local_seq = 0
        book_state: Optional[_BookState] = None

        async for payload in self._resilient_stream(channel="book", symbols=[pair], depth=depth, snapshot=True):
            if not isinstance(payload, dict):
                continue
            if payload.get("channel") != "book" or "data" not in payload:
                continue

            # Kraken v2 always sends data as a list; first item is the book object
            book_msg = payload["data"][0]
            symbol = str(book_msg["symbol"])
            bids = _levels_from_v2(book_msg.get("bids", []))
            asks = _levels_from_v2(book_msg.get("asks", []))
            checksum = book_msg.get("checksum")

            if payload.get("type") == "snapshot":
                book_state = _BookState.from_snapshot(book_msg.get("asks", []), book_msg.get("bids", []), depth)
                local_seq += 1
                yield BookMessage(
                    ts=utc_now_iso(),
                    symbol=symbol,
                    type="snapshot",
                    seq=local_seq,
                    bids=bids,
                    asks=asks,
                )
            elif payload.get("type") == "update":
                local_seq += 1
                if book_state is not None:
                    # Use raw dicts to update (string-safe)
                    book_state.apply_update(book_msg.get("asks", []), book_msg.get("bids", []))
                    if checksum is not None:
                        calc = _compute_checksum_v2(book_state.top10_asks(), book_state.top10_bids())
                        if calc != str(checksum):
                            raise ResyncRequired(f"Checksum mismatch (recv={checksum}, calc={calc})")
                yield BookMessage(
                    ts=str(book_msg.get("timestamp", utc_now_iso())),
                    symbol=symbol,
                    type="delta",
                    seq=local_seq,
                    bids=bids,
                    asks=asks,
                )
            else:
                # Unknown type: ignore
                continue

    # ---- resilient stream ----
    async def _resilient_stream(self, *, channel: str, symbols: List[str], **kwargs: Any) -> AsyncIterator[Any]:
        delays = backoff_schedule()
        ping_task: Optional[asyncio.Task] = None
        latencies_ms: List[float] = []
        last_recv = time.perf_counter()

        while True:
            try:
                async with await self._connect() as ws:
                    await self._subscribe(ws, channel, symbols=symbols, **kwargs)

                    async def _pinger() -> None:
                        while True:
                            await asyncio.sleep(20.0)
                            with contextlib.suppress(Exception):
                                await self._ping(ws)

                    ping_task = asyncio.create_task(_pinger())

                    while True:
                        msg = await self._recv_json(ws)
                        now = time.perf_counter()
                        latencies_ms.append((now - last_recv) * 1000.0)
                        last_recv = now

                        # System/admin messages to ignore (status/heartbeat/pong/subscribe acks)
                        if isinstance(msg, dict):
                            if msg.get("method") in ("pong", "subscribe", "unsubscribe"):
                                self.logger.debug({"event": f"ws.{msg.get('method')}", "ts": utc_now_iso()})
                                continue
                            ch = msg.get("channel")
                            if ch in ("status", "heartbeat"):
                                self.logger.debug({"event": f"ws.{ch}", "ts": utc_now_iso()})
                                continue
                        yield msg
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Log p50/p95 since last connect
                p50 = _pctl(latencies_ms, 50)
                p95 = _pctl(latencies_ms, 95)
                self.logger.error({
                    "event": "ws.disconnected",
                    "ts": utc_now_iso(),
                    "error": str(e),
                    "p50_ms": p50,
                    "p95_ms": p95,
                    "session": self.session_id,
                })
                delay = await delays.__anext__()  # capped < 5s
                await asyncio.sleep(delay)
            finally:
                if ping_task:
                    ping_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await ping_task


def _pctl(x: List[float], p: float) -> float:
    if not x:
        return 0.0
    try:
        import numpy as np
        return float(np.percentile(x, p))
    except Exception:
        return 0.0
