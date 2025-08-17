from __future__ import annotations

"""
CLI principal (Kraken WS v2) — fonctionne à la fois:
- comme module:      python -m crypto_mm.main ws-trades --pair BTC/USD
- comme script brut: python crypto_mm/crypto_mm/main.py ws-trades --pair BTC/USD

Sous-commandes utiles:
- check                : sanity offline
- ws-ping              : ping v2
- ws-trades            : stream trades (TTY 500ms)
- ws-book              : stream L2 (top best bid/ask via messages)
- ws-book-ladder       : (optionnel via CRYPTO_MM_EXTRAS=1) L2 stateful
- ws-live              : (optionnel via CRYPTO_MM_EXTRAS=1) live TTY book+trades
- live/replay/plot     : démos utilitaires
"""

import argparse
import asyncio
import contextlib
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

# --- imports robustes (module OU script brut) ---
try:
    # Exécution comme module: imports relatifs OK
    from .core.config import Settings, load_settings
    from .core.clock import Clock
    from .core.log import get_logger
    from .core.types import utc_now_iso
    from .data.ws_client import KrakenWSV2Client, ResyncRequired
    from .data.order_book import OrderBookL2
    from .ui.live_view import LiveState, render, stop_render, log_event
except Exception:  # pragma: no cover - fallback pour exécution directe
    import pathlib

    PKG_ROOT = str(pathlib.Path(__file__).resolve().parents[1])  # répertoire parent contenant le package 'crypto_mm'
    if PKG_ROOT not in sys.path:
        sys.path.insert(0, PKG_ROOT)

    from crypto_mm.core.config import Settings, load_settings  # type: ignore
    from crypto_mm.core.clock import Clock  # type: ignore
    from crypto_mm.core.log import get_logger  # type: ignore
    from crypto_mm.core.types import utc_now_iso  # type: ignore
    from crypto_mm.data.ws_client import KrakenWSV2Client, ResyncRequired  # type: ignore
    from crypto_mm.data.order_book import OrderBookL2  # type: ignore
    from crypto_mm.ui.live_view import LiveState, render, stop_render, log_event  # type: ignore


# ---------- Parser ----------

def build_parser(*, extras: Optional[bool] = False) -> argparse.ArgumentParser:
    """
    Construire le parser CLI.

    Paramètres
    ----------
    extras : bool, optional
        Si True, expose aussi `ws-book-ladder` et `ws-live`.
        Par défaut False pour garantir un set stable pour les tests.
    """
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

    # --- Extras optionnels (pour ne pas casser les tests qui vérifient l'égalité exacte) ---
    if extras:
        p_ladder = sub.add_parser("ws-book-ladder", help="Stateful L2: mid/microprice + ladder filtered.")
        p_ladder.add_argument("--pair", type=str, default="BTC/USD", help="Kraken pair (e.g., BTC/USD).")
        p_ladder.add_argument("--depth", type=int, default=10, help="Depth subscription (10/25/100/500/1000).")
        p_ladder.add_argument("--min-size", type=float, default=0.0, help="Filter out levels with qty < min_size.")
        p_ladder.add_argument("--duration", type=float, default=10.0, help="Duration in seconds.")

        p_livews = sub.add_parser("ws-live", help="Live TTY view: book (stateful) + trades (tape).")
        p_livews.add_argument("--pair", type=str, default="BTC/USD")
        p_livews.add_argument("--depth", type=int, default=10)
        p_livews.add_argument("--min-size", type=float, default=0.0)
        p_livews.add_argument("--duration", type=float, default=20.0)

    return parser


def _apply_config_path_arg(cfg_path: Optional[str]) -> None:
    if cfg_path:
        os.environ["CRYPTO_MM_CONFIG"] = cfg_path


# ---------- Démos utilitaires ----------

def cmd_live(settings: Settings, duration: float) -> int:
    logger = get_logger("live", settings.log_level)
    clock = Clock(period_ms=settings.heartbeat_ms, logger=logger)
    logger.info({"event": "live.start", "ts": utc_now_iso(), "heartbeat_ms": settings.heartbeat_ms})

    start = time.perf_counter()

    def on_tick(tick: int, dt_ms: float) -> None:
        elapsed = time.perf_counter() - start
        msg = f"\r[{utc_now_iso()}] LIVE tick={tick} elapsed={elapsed:0.2f}s dt={dt_ms:0.1f}ms p50={clock.p50_ms():0.1f}ms p95={clock.p95_ms():0.1f}ms"
        sys.stdout.write(msg)
        sys.stdout.flush()

    clock.run(duration_s=duration, iterations=None, on_tick=on_tick)
    sys.stdout.write("\n")
    logger.info(clock.metrics_summary_event("live.stop"))
    return 0


def cmd_replay(settings: Settings, duration: float, speed: float) -> int:
    logger = get_logger("replay", settings.log_level)
    clock = Clock(period_ms=int(settings.heartbeat_ms / max(speed, 0.1)), logger=logger)
    logger.info({"event": "replay.start", "ts": utc_now_iso(), "heartbeat_ms": clock.period_ms, "speed": speed})

    start = time.perf_counter()

    def on_tick(tick: int, dt_ms: float) -> None:
        elapsed = time.perf_counter() - start
        msg = f"\r[{utc_now_iso()}] REPLAY tick={tick} elapsed={elapsed:0.2f}s dt={dt_ms:0.1f}ms p50={clock.p50_ms():0.1f}ms p95={clock.p95_ms():0.1f}ms"
        sys.stdout.write(msg)
        sys.stdout.flush()

    clock.run(duration_s=duration, iterations=None, on_tick=on_tick)
    sys.stdout.write("\n")
    logger.info(clock.metrics_summary_event("replay.stop"))
    return 0


def cmd_plot(settings: Settings, output: Optional[str]) -> int:
    import matplotlib.pyplot as plt
    logger = get_logger("plot", settings.log_level)
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
    logger.info({"event": "plot.saved", "ts": utc_now_iso(), "output": out})
    print(f"Plot saved to: {out}")
    return 0


def cmd_check(settings: Settings) -> int:
    """Lightweight checks: imports, clock tick, offline checksum/mappers."""
    try:
        # Cas package
        from .data.ws_client import _BookState, _compute_checksum_v2, _levels_from_v2  # type: ignore
    except Exception:
        # Cas script direct
        from crypto_mm.data.ws_client import _BookState, _compute_checksum_v2, _levels_from_v2  # type: ignore

    logger = get_logger("check", settings.log_level)
    # 1) Clock quick tick
    clock = Clock(period_ms=100, logger=logger)
    ticks = {"n": 0}

    def cb(_i: int, _dt: float) -> None:
        ticks["n"] += 1

    clock.run(duration_s=0.4, iterations=None, on_tick=cb)

    # 2) Offline book snapshot+update and checksum v2
    asks = [{"price": "50000.0", "qty": "1.0"}, {"price": "50010.0", "qty": "2.0"}]
    bids = [{"price": "49990.0", "qty": "1.5"}, {"price": "49980.0", "qty": "3.0"}]
    st = _BookState.from_snapshot(asks, bids, depth=10)
    up_a = [{"price": "50005.0", "qty": "0.5"}]
    up_b = [{"price": "49995.0", "qty": "0.0"}]
    st.apply_update(up_a, up_b)
    _ = _levels_from_v2(up_a + up_b)
    _ = _compute_checksum_v2(st.top10_asks(), st.top10_bids())

    print("CHECK OK: imports, clock, v2 mapping+checksum.")
    return 0


# ---------- WS helpers ----------

async def _ws_ping_async(settings: Settings) -> int:
    client = KrakenWSV2Client(settings=settings, logger=get_logger("ws-ping", settings.log_level))
    # Ouvre un flux et quitte dès que quelque chose arrive (ping implicite par subscribe)
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
                    f"\r[{last.ts}] {pair} TRADE id={last.trade_id} side={last.side} px={last.price_usd_per_btc:.2f} qty={last.qty_btc:.6f}  p50={np.percentile(latencies,50):.1f}ms p95={np.percentile(latencies,95):.1f}ms"
                )
            sys.stdout.flush()
        sys.stdout.write("\n")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    logger.info({"event": "ws-trades.stop", "ts": utc_now_iso()})
    return 0


async def _ws_book_async(settings: Settings, pair: str, duration: float, depth: int) -> int:
    logger = get_logger("ws-book", settings.log_level)
    client = KrakenWSV2Client(settings=settings, logger=logger)

    last_seq: Optional[int] = None
    best_bid: Optional[Tuple[float, float]] = None
    best_ask: Optional[Tuple[float, float]] = None

    latencies: List[float] = []
    last_print = time.perf_counter()
    start = time.perf_counter()

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
    try:
        while (time.perf_counter() - start) < duration:
            await asyncio.sleep(settings.heartbeat_ms / 1000.0)
            now = time.perf_counter()
            latencies.append((now - last_print) * 1000.0)
            last_print = now
            if last_seq is None:
                sys.stdout.write(f"\r[{utc_now_iso()}] {pair} waiting for snapshot/deltas...")
            else:
                bb = f"{best_bid[0]:.2f}/{best_bid[1]:.4f}" if best_bid else "NA"
                aa = f"{best_ask[0]:.2f}/{best_ask[1]:.4f}" if best_ask else "NA"
                sys.stdout.write(
                    f"\r[{utc_now_iso()}] {pair} SEQ={last_seq} BEST_BID px/qty={bb}  BEST_ASK px/qty={aa}  p50={np.percentile(latencies,50):.1f}ms p95={np.percentile(latencies,95):.1f}ms"
                )
            sys.stdout.flush()
        sys.stdout.write("\n")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    logger.info({"event": "ws-book.stop", "ts": utc_now_iso()})
    return 0


async def _ws_book_ladder_async(settings: Settings, pair: str, duration: float, depth: int, min_size: float) -> int:
    """
    Maintient un carnet L2 stateful (OrderBookL2) et affiche mid/microprice + ladder filtré.
    """
    logger = get_logger("ws-book-ladder", settings.log_level)
    client = KrakenWSV2Client(settings=settings, logger=logger)
    ob = OrderBookL2(logger=logger)

    start = time.perf_counter()
    last_print = time.perf_counter()
    latencies: List[float] = []

    async def collector() -> None:
        # Reboucle en cas de resync
        while True:
            try:
                async for msg in client.subscribe_order_book(pair, depth=depth):
                    if msg.type == "snapshot":
                        ob.apply_snapshot(msg)
                    else:
                        ob.apply_delta(msg)
            except ResyncRequired as e:
                logger.error({"event": "ws-book-ladder.resync", "ts": utc_now_iso(), "error": str(e)})
                ob.clear()
                continue  # repart sur un nouveau subscribe

    task = asyncio.create_task(collector())
    try:
        while (time.perf_counter() - start) < duration:
            await asyncio.sleep(settings.heartbeat_ms / 1000.0)
            now = time.perf_counter()
            latencies.append((now - last_print) * 1000.0)
            last_print = now

            top = ob.top(1)
            best_bid = top["bids"][0] if top["bids"] else None
            best_ask = top["asks"][0] if top["asks"] else None
            mid = ob.mid_price()
            micro = ob.microprice()
            ladder = ob.ladder(depth=depth, min_size=min_size)

            bb = f"{best_bid[0]:.2f}/{best_bid[1]:.4f}" if best_bid else "NA"
            aa = f"{best_ask[0]:.2f}/{best_ask[1]:.4f}" if best_ask else "NA"
            mid_s = f"{mid:.2f}" if mid == mid else "NA"
            micro_s = f"{micro:.2f}" if micro == micro else "NA"

            # résumé ladder compact (top 3 par côté)
            ladder_bids = " ".join([f"{p:.0f}@{q:.3f}" for p, q in ladder["bids"][:3]]) or "-"
            ladder_asks = " ".join([f"{p:.0f}@{q:.3f}" for p, q in ladder["asks"][:3]]) or "-"

            sys.stdout.write(
                f"\r[{utc_now_iso()}] {pair} SEQ={ob.last_seq} "
                f"BEST_BID={bb} BEST_ASK={aa} MID={mid_s} MICRO={micro_s} "
                f"BIDS({len(ladder['bids'])}): {ladder_bids}  ASKS({len(ladder['asks'])}): {ladder_asks}  "
                f"p50={np.percentile(latencies,50):.1f}ms p95={np.percentile(latencies,95):.1f}ms"
            )
            sys.stdout.flush()
        sys.stdout.write("\n")
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    logger.info(ob.metrics_event("ws-book-ladder.stop"))
    return 0


async def _ws_live_async(settings: Settings, pair: str, duration: float, depth: int, min_size: float) -> int:
    """
    Vue TTY complète (book stateful + trade tape) rafraîchie toutes heartbeat_ms (~500 ms).
    Affiche le book à gauche et un event-log à droite (snapshots, resync, derniers trades).
    """
    logger = get_logger("ws-live", settings.log_level)

    # Taisons les logs verbeux dans la console : on garde stderr pour erreurs
    import logging
    for h in list(logger.handlers):
        if getattr(h, "stream", None) is sys.stdout:
            h.stream = sys.stderr
    logger.setLevel(logging.ERROR)

    client = KrakenWSV2Client(settings=settings, logger=logger)

    # Crée des loggers "quiet" pour les sous-composants (évite les JSON qui polluent l'UI)
    ob_logger = get_logger("ws-live.ob", settings.log_level)
    tape_logger = get_logger("ws-live.tape", settings.log_level)
    for lg in (ob_logger, tape_logger):
        for h in list(lg.handlers):
            if getattr(h, "stream", None) is sys.stdout:
                h.stream = sys.stderr
        lg.setLevel(logging.ERROR)

    ob = OrderBookL2(logger=ob_logger)
    from crypto_mm.data.trade_tape import TradeTape  # import local
    tape = TradeTape(capacity=2000, logger=tape_logger)
    state = LiveState(pair=pair, order_book=ob, trade_tape=tape, depth=depth, min_size=min_size)

    log_event("ws.connected")

    async def book_task():
        while True:
            try:
                async for msg in client.subscribe_order_book(pair, depth=depth):
                    if msg.type == "snapshot":
                        ob.apply_snapshot(msg)
                        log_event(f"snapshot seq={msg.seq}")
                    else:
                        ob.apply_delta(msg)
                        # on ne spam pas chaque delta; on peut échantillonner…
                # fin du for
            except ResyncRequired as e:
                log_event(f"RESYNC required: {e}")
                ob.clear()
                continue

    async def trades_task():
        async for tr in client.subscribe_trades(pair):
            tape.push(tr)
            # le render() affichera le dernier trade en event automatiquement

    t1 = asyncio.create_task(book_task())
    t2 = asyncio.create_task(trades_task())

    start = time.perf_counter()
    try:
        while (time.perf_counter() - start) < duration:
            await asyncio.sleep(settings.heartbeat_ms / 1000.0)
            render(state, out=sys.stdout)  # mise à jour incrémentale (pas d’effacement total)
    except KeyboardInterrupt:
        log_event("KeyboardInterrupt")
    finally:
        for t in (t1, t2):
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t
        with contextlib.suppress(Exception):
            stop_render()
    return 0

# ---------- Entrées CLI ----------

def cmd_check_all(settings: Settings) -> int:
    return cmd_check(settings)


def cmd_ws_ping(settings: Settings) -> int:
    return asyncio.run(_ws_ping_async(settings))


def cmd_ws_trades(settings: Settings, pair: str, duration: float) -> int:
    return asyncio.run(_ws_trades_async(settings, pair, duration))


def cmd_ws_book(settings: Settings, pair: str, duration: float, depth: int) -> int:
    return asyncio.run(_ws_book_async(settings, pair, duration, depth))


def cmd_ws_book_ladder(settings: Settings, pair: str, duration: float, depth: int, min_size: float) -> int:
    return asyncio.run(_ws_book_ladder_async(settings, pair, duration, depth, min_size))


def cmd_ws_live(settings: Settings, pair: str, duration: float, depth: int, min_size: float) -> int:
    return asyncio.run(_ws_live_async(settings, pair, duration, depth, min_size))


def main(argv: Optional[list[str]] = None) -> int:
    # Pour le binaire/CLI, on décide ici si on expose les extras en se basant sur l'env,
    # tout en gardant build_parser() par défaut sans extras pour les tests unitaires.
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
        return cmd_ws_book(settings, pair=args.pair, duration=args.duration, depth=args.depth)
    if args.cmd == "ws-book-ladder":
        return cmd_ws_book_ladder(settings, pair=args.pair, duration=args.duration, depth=args.depth, min_size=args.min_size)
    if args.cmd == "ws-live":
        return cmd_ws_live(settings, pair=args.pair, duration=args.duration, depth=args.depth, min_size=args.min_size)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
