from __future__ import annotations

"""
CLI principal (Kraken WS v2):

Sous-commandes de base :
- live / replay / plot / check / ws-ping / ws-trades / ws-book

Extras (CRYPTO_MM_EXTRAS=1) :
- ws-book-ladder
- ws-live  (UI TTY : OrderBook + KPIs + Events)

UI : rafraîchissement ~500 ms (paramétré par heartbeat_ms).
"""

import argparse
import asyncio
import contextlib
import os
import sys
import time
import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np

# --- imports robustes (module OU script brut) ---
try:
    from .core.config import Settings, load_settings
    from .core.clock import Clock
    from .core.log import get_logger
    from .core.types import utc_now_iso
    from .data.ws_client import KrakenWSV2Client, ResyncRequired
    from .data.order_book import OrderBookL2
    from .ui.live_view import LiveState, RenderOptions, render, stop_render, log_event
except Exception:  # pragma: no cover
    import pathlib

    PKG_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
    if PKG_ROOT not in sys.path:
        sys.path.insert(0, PKG_ROOT)

    from crypto_mm.core.config import Settings, load_settings  # type: ignore
    from crypto_mm.core.clock import Clock  # type: ignore
    from crypto_mm.core.log import get_logger  # type: ignore
    from crypto_mm.core.types import utc_now_iso  # type: ignore
    from crypto_mm.data.ws_client import KrakenWSV2Client, ResyncRequired  # type: ignore
    from crypto_mm.data.order_book import OrderBookL2  # type: ignore
    from crypto_mm.ui.live_view import LiveState, RenderOptions, render, stop_render, log_event  # type: ignore


# ---------- Utils ----------

def _apply_config_path_arg(cfg_path: Optional[str]) -> None:
    if cfg_path:
        os.environ["CRYPTO_MM_CONFIG"] = cfg_path


# Context manager : coupe TOUT logging vers la console pendant ws-live
import logging
from contextlib import contextmanager

@contextmanager
def _squelch_console_logging():
    """
    Désactive les logs (< CRITICAL) et redirige les StreamHandlers stdout->stderr.
    Supprime aussi les warnings (FutureWarning, etc.) pendant l'UI.
    """
    prev_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)

    # warnings
    warnings_filters_backup = warnings.filters[:]
    warnings.simplefilter("ignore")

    modified: list[tuple[logging.Logger, logging.Handler, object]] = []
    try:
        all_names = [""] + [n for n in logging.root.manager.loggerDict.keys()]  # type: ignore[attr-defined]
        for name in all_names:
            lg = logging.getLogger(name)
            for h in list(getattr(lg, "handlers", [])):
                if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) in (sys.stdout, sys.__stdout__):
                    modified.append((lg, h, h.stream))
                    try: h.flush()
                    except Exception: pass
                    h.stream = sys.stderr
        yield
    finally:
        # restore handlers
        for lg, h, old_stream in modified:
            try: h.flush()
            except Exception: pass
            h.stream = old_stream
        # restore logging / warnings
        logging.disable(prev_disable)
        warnings.filters[:] = warnings_filters_backup

# ---------- Parser ----------
def build_parser(*, extras: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crypto_mm",
        description="Crypto MM scaffold (Kraken WS v2) with 500ms heartbeat and CLI demos.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (default: ./config.yaml)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_live = sub.add_parser("live", help="Run live loop (demo; no external API).")
    p_live.add_argument("--duration", type=float, default=5.0, help="Duration in seconds (demo).")

    p_replay = sub.add_parser("replay", help="Run replay loop (demo).")
    p_replay.add_argument("--duration", type=float, default=5.0, help="Duration in seconds (demo).")
    p_replay.add_argument("--speed", type=float, default=1.0, help="Playback speed factor.")

    p_plot = sub.add_parser("plot", help="Produce a demo plot (matplotlib).")
    p_plot.add_argument("--output", type=str, default=None, help="Output file (PNG).")

    # --- Self-checks & WS ---
    sub.add_parser("check", help="Run quick checks (imports, clock, offline mapping).")
    sub.add_parser("ws-ping", help="Send a ping event to Kraken v2 and await first message.")

    p_tr = sub.add_parser("ws-trades", help="Stream Kraken trades (TTY refresh 500ms).")
    p_tr.add_argument("--pair", type=str, default="BTC/USD", help="Kraken pair (e.g., BTC/USD).")
    p_tr.add_argument("--duration", type=float, default=5.0, help="Duration in seconds.")

    p_ob = sub.add_parser("ws-book", help="Stream Kraken order book (snapshot + deltas + checksum).")
    p_ob.add_argument("--pair", type=str, default="BTC/USD", help="Kraken pair (e.g., BTC/USD).")
    p_ob.add_argument("--depth", type=int, default=10, help="Depth (10/25/100/500/1000).")
    p_ob.add_argument("--duration", type=float, default=5.0, help="Duration in seconds.")

    # --- Extras optionnels (désactivés par défaut pour les tests) ---
    if extras:
        p_ladder = sub.add_parser("ws-book-ladder", help="Stateful L2: mid/microprice + ladder filtered.")
        p_ladder.add_argument("--pair", type=str, default="BTC/USD")
        p_ladder.add_argument("--depth", type=int, default=10)
        p_ladder.add_argument("--min-size", type=float, default=0.0)
        p_ladder.add_argument("--duration", type=float, default=10.0)

        p_livews = sub.add_parser("ws-live", help="Live TTY view: book (stateful) + trades (tape).")
        p_livews.add_argument("--pair", type=str, default="BTC/USD")
        p_livews.add_argument("--depth", type=int, default=10)
        p_livews.add_argument("--min-size", type=float, default=0.0)
        p_livews.add_argument("--duration", type=float, default=20.0)
        p_livews.add_argument("--events", action="store_true", help="Show right-side events panel.")

    return parser



# ---------- Démos utilitaires (sans WS) ----------

def cmd_live(settings: Settings, duration: float) -> int:
    clock = Clock(period_ms=settings.heartbeat_ms, logger=get_logger("live", settings.log_level))
    start = time.perf_counter()

    def on_tick(tick: int, dt_ms: float) -> None:
        elapsed = time.perf_counter() - start
        msg = f"\r[{utc_now_iso()}] LIVE tick={tick} elapsed={elapsed:0.2f}s dt={dt_ms:0.1f}ms p50={clock.p50_ms():0.1f}ms p95={clock.p95_ms():0.1f}ms"
        sys.stdout.write(msg)
        sys.stdout.flush()

    clock.run(duration_s=duration, iterations=None, on_tick=on_tick)
    sys.stdout.write("\n")
    return 0


def cmd_replay(settings: Settings, duration: float, speed: float) -> int:
    clock = Clock(period_ms=int(settings.heartbeat_ms / max(speed, 0.1)), logger=get_logger("replay", settings.log_level))
    start = time.perf_counter()

    def on_tick(tick: int, dt_ms: float) -> None:
        elapsed = time.perf_counter() - start
        msg = f"\r[{utc_now_iso()}] REPLAY tick={tick} elapsed={elapsed:0.2f}s dt={dt_ms:0.1f}ms p50={clock.p50_ms():0.1f}ms p95={clock.p95_ms():0.1f}ms"
        sys.stdout.write(msg)
        sys.stdout.flush()

    clock.run(duration_s=duration, iterations=None, on_tick=on_tick)
    sys.stdout.write("\n")
    return 0


def cmd_plot(settings: Settings, output: Optional[str]) -> int:
    import matplotlib.pyplot as plt

    out = output or os.path.join(settings.plot_output_dir, f"latency_{int(time.time())}.png")
    x = np.arange(100)
    y = np.sin(x / 10.0)
    plt.figure()
    plt.plot(x, y)
    plt.title("Demo Plot")
    plt.xlabel("Index")
    plt.ylabel("Value")
    os.makedirs(settings.plot_output_dir, exist_ok=True)
    plt.savefig(out)
    print(f"Plot saved to: {out}")
    return 0


def cmd_check(settings: Settings) -> int:
    from .data.ws_client import _BookState, _compute_checksum_v2  # type: ignore

    clock = Clock(period_ms=100, logger=get_logger("check", settings.log_level))
    ticks = {"n": 0}

    def cb(_i: int, _dt: float) -> None:
        ticks["n"] += 1

    clock.run(duration_s=0.4, iterations=None, on_tick=cb)

    asks = [{"price": "50000.0", "qty": "1.0"}]
    bids = [{"price": "49990.0", "qty": "1.5"}]
    st = _BookState.from_snapshot(asks, bids, depth=10)
    _ = _compute_checksum_v2(st.top10_asks(), st.top10_bids())
    print("CHECK OK")
    return 0


# ---------- WS helpers ----------

async def _ws_ping_async(settings: Settings) -> int:
    client = KrakenWSV2Client(settings=settings, logger=get_logger("ws-ping", settings.log_level))
    async for _ in client._resilient_stream(channel="trade", symbols=["BTC/USD"]):
        return 0
    return 1


async def _ws_trades_async(settings: Settings, pair: str, duration: float) -> int:
    logger = get_logger("ws-trades", settings.log_level)
    client = KrakenWSV2Client(settings=settings, logger=logger)

    last = None
    latencies: List[float] = []
    last_print = time.perf_counter()
    start = time.perf_counter()

    async def collector() -> None:
        nonlocal last
        async for t in client.subscribe_trades(pair):
            last = t

    task = asyncio.create_task(collector())
    try:
        while (time.perf_counter() - start) < duration:
            await asyncio.sleep(settings.heartbeat_ms / 1000.0)
            now = time.perf_counter()
            latencies.append((now - last_print) * 1000.0)
            last_print = now
            if last is None:
                sys.stdout.write(f"\r[{utc_now_iso()}] {pair} waiting for trades...")
            else:
                sys.stdout.write(
                    f"\r[{last.ts}] {pair} TRADE id={last.trade_id} {last.side} "
                    f"px={last.price_usd_per_btc:.2f} qty={last.qty_btc:.6f}  "
                    f"p50={np.percentile(latencies,50):.1f}ms p95={np.percentile(latencies,95):.1f}ms"
                )
            sys.stdout.flush()
        sys.stdout.write("\n")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return 0


async def _ws_book_async(settings: Settings, pair: str, duration: float, depth: int) -> int:
    logger = get_logger("ws-book", settings.log_level)
    client = KrakenWSV2Client(settings=settings, logger=logger)

    last_seq: Optional[int] = None
    best_bid: Optional[Tuple[float, float]] = None
    best_ask: Optional[Tuple[float, float]] = None

    async def collector() -> None:
        nonlocal last_seq, best_bid, best_ask
        try:
            async for msg in client.subscribe_order_book(pair, depth=depth):
                last_seq = msg.seq

                def top(levels):
                    for lvl in levels:
                        if lvl.action == "upsert":
                            return (lvl.price_usd_per_btc, lvl.qty_btc)
                    return None

                b = top(msg.bids)
                a = top(msg.asks)
                if b:
                    best_bid = b
                if a:
                    best_ask = a
        except ResyncRequired as e:
            logger.error({"event": "ws-book.resync", "ts": utc_now_iso(), "error": str(e)})

    task = asyncio.create_task(collector())
    start = time.perf_counter()
    try:
        while (time.perf_counter() - start) < duration:
            await asyncio.sleep(settings.heartbeat_ms / 1000.0)
            if last_seq is None:
                sys.stdout.write(f"\r[{utc_now_iso()}] {pair} waiting for snapshot/deltas...")
            else:
                bb = f"{best_bid[0]:.2f}/{best_bid[1]:.4f}" if best_bid else "NA"
                aa = f"{best_ask[0]:.2f}/{best_ask[1]:.4f}" if best_ask else "NA"
                sys.stdout.write(f"\r[{utc_now_iso()}] {pair} SEQ={last_seq} BID {bb}  ASK {aa}")
            sys.stdout.flush()
        sys.stdout.write("\n")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return 0


# ---------- WS: Live UI (extras) ----------

async def _ws_live_async(
    settings: Settings,
    pair: str,
    duration: float,
    depth: int,
    min_size: float,
) -> int:
    """
    Vue TTY complète (book stateful + trade tape) rafraîchie toutes heartbeat_ms (~500 ms).
    Panneau gauche : OrderBook + KPIs + ladder summary
    Panneau droit  : Events (tsZ | type | side | qty | price)
    """
    from crypto_mm.data.trade_tape import TradeTape  # import local

    # **Bloque à la source** toute écriture logging qui polluerait la TTY :
    with _squelch_console_logging():
        client = KrakenWSV2Client(settings=settings, logger=get_logger("ws-live", settings.log_level))
        ob = OrderBookL2(logger=get_logger("ws-live.ob", settings.log_level))
        tape = TradeTape(capacity=2000, logger=get_logger("ws-live.tape", settings.log_level))

        state = LiveState(pair=pair, order_book=ob, trade_tape=tape, depth=depth, min_size=min_size)
        opts = RenderOptions(color=True)

        log_event("startup")

        async def book_task() -> None:
            while True:
                try:
                    async for msg in client.subscribe_order_book(pair, depth=depth):
                        if msg.type == "snapshot":
                            ob.apply_snapshot(msg)
                            log_event("snapshot")
                        else:
                            ob.apply_delta(msg)
                except ResyncRequired:
                    log_event("resync")
                    ob.clear()
                    continue

        async def trades_task() -> None:
            async for tr in client.subscribe_trades(pair):
                tape.push(tr)
                # >>> NEW: évènement structuré pour l'UI (best-effort)
                try:
                    log_event(f"trade {tr.side} {tr.qty_btc:.6f} @ {tr.price_usd_per_btc:.2f}")
                except Exception:
                    pass

        t1 = asyncio.create_task(book_task())
        t2 = asyncio.create_task(trades_task())

        start = time.perf_counter()
        try:
            while (time.perf_counter() - start) < duration:
                await asyncio.sleep(settings.heartbeat_ms / 1000.0)
                render(state, opts=opts)  # mise à jour in-place, une seule fenêtre
        except KeyboardInterrupt:
            pass
        finally:
            stop_render()
            for t in (t1, t2):
                t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(t1, t2)

    return 0


def cmd_ws_ping(settings: Settings) -> int:
    return asyncio.run(_ws_ping_async(settings))


def cmd_ws_trades(settings: Settings, pair: str, duration: float) -> int:
    return asyncio.run(_ws_trades_async(settings, pair, duration))


def cmd_ws_book(settings: Settings, pair: str, duration: float, depth: int) -> int:
    return asyncio.run(_ws_book_async(settings, pair, duration, depth))


def cmd_ws_live(settings: Settings, pair: str, duration: float, depth: int, min_size: float) -> int:
    return asyncio.run(_ws_live_async(settings, pair, duration, depth, min_size))

def cmd_check_all(settings: Settings) -> int:
    return cmd_check(settings)

# Optionnel : ws-book-ladder (si extras)
async def _ws_book_ladder_async(settings: Settings, pair: str, duration: float, depth: int, min_size: float) -> int:
    logger = get_logger("ws-book-ladder", settings.log_level)
    client = KrakenWSV2Client(settings=settings, logger=logger)
    ob = OrderBookL2(logger=logger)
    start = time.perf_counter()

    async def collector() -> None:
        while True:
            try:
                async for msg in client.subscribe_order_book(pair, depth=depth):
                    if msg.type == "snapshot":
                        ob.apply_snapshot(msg)
                    else:
                        ob.apply_delta(msg)
            except ResyncRequired:
                ob.clear()
                continue

    task = asyncio.create_task(collector())
    try:
        while (time.perf_counter() - start) < duration:
            await asyncio.sleep(settings.heartbeat_ms / 1000.0)
            top = ob.top(1)
            bb = top["bids"][0] if top["bids"] else None
            aa = top["asks"][0] if top["asks"] else None
            mid = ob.mid_price()
            micro = ob.microprice()
            sys.stdout.write(
                f"\r{pair} mid={mid:.2f} micro={micro:.2f} "
                f"best_bid={bb[0]:.2f}/{bb[1]:.4f} best_ask={aa[0]:.2f}/{aa[1]:.4f}      "
            ); sys.stdout.flush()
        sys.stdout.write("\n")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return 0

def cmd_ws_book_ladder(settings: Settings, pair: str, duration: float, depth: int, min_size: float) -> int:
    return asyncio.run(_ws_book_ladder_async(settings, pair, duration, depth, min_size))





# ---------- Entrée CLI ----------

def main(argv: Optional[list[str]] = None) -> int:
    # Extras activables uniquement depuis l'ENV pour l'exécutable,
    # mais PAS pris en compte par tests qui appellent build_parser() sans arg.
    extras_env = os.environ.get("CRYPTO_MM_EXTRAS", "0").lower() in {"1", "true", "yes", "on"}
    parser = build_parser(extras=extras_env)

    args = parser.parse_args(argv)
    if args.config:
        os.environ["CRYPTO_MM_CONFIG"] = args.config

    settings = load_settings()

    if args.cmd == "live":
        return cmd_live(settings, duration=args.duration)
    if args.cmd == "replay":
        return cmd_replay(settings, duration=args.duration, speed=args.speed)
    if args.cmd == "plot":
        return cmd_plot(settings, output=args.output)
    if args.cmd == "check":
        return cmd_check_all(settings)
    if args.cmd == "ws-ping":
        return cmd_ws_ping(settings)
    if args.cmd == "ws-trades":
        return cmd_ws_trades(settings, pair=args.pair, duration=args.duration)
    if args.cmd == "ws-book":
        return cmd_ws_book(settings, pair=args.pair, duration=args.duration)

    # Extras
    if extras_env and args.cmd == "ws-book-ladder":
        return cmd_ws_book_ladder(settings, pair=args.pair, duration=args.duration, depth=args.depth, min_size=args.min_size)
    if extras_env and args.cmd == "ws-live":
        return cmd_ws_live(settings, pair=args.pair, duration=args.duration, depth=args.depth, min_size=args.min_size)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())



