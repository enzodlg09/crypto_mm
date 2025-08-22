from __future__ import annotations
"""
CLI principal (Kraken WS v2)

Sous-commandes de base :
- live / replay / plot / check / ws-ping / ws-trades / ws-book

Extras (CRYPTO_MM_EXTRAS=1) :
- ws-book-ladder
- ws-live  (UI TTY : OrderBook + KPIs + Last trades)
- sim-live (E2E : WS -> KPIs -> quotes -> risk -> exec -> inventory -> UI + CSV)

UI : rafraîchissement ~500 ms (paramétré par heartbeat_ms).
Timestamps : UTC ISO8601 (suffixe 'Z')
Quantités : BTC, Prix : USD/BTC, Montants : USD
"""

import argparse
import asyncio
import contextlib
import csv
import logging
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import math

# --- imports robustes (module OU exécution directe) ---
try:
    from .core.config import Settings, load_settings
    from .core.clock import Clock
    from .core.log import get_logger
    from .core.types import utc_now_iso
    from .data.ws_client import KrakenWSV2Client, ResyncRequired
    from .data.order_book import OrderBookL2
    from .ui.live_view import LiveState, RenderOptions, render, stop_render
    # E2E (sim-live)
    from .mm.spread_kpi import KpiStore, spread_for_size
    from .mm.quoting import generate_quote
    from .risk.limits import RiskGate, Decision
    from .sim.execution import simulate_fills, QuotePair
    from .mm.inventory import Inventory
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
    from crypto_mm.ui.live_view import LiveState, RenderOptions, render, stop_render, UiFill  # type: ignore
    # E2E (sim-live)
    from crypto_mm.mm.spread_kpi import KpiStore, spread_for_size  # type: ignore
    from crypto_mm.mm.quoting import generate_quote  # type: ignore
    from crypto_mm.risk.limits import RiskGate, Decision  # type: ignore
    from crypto_mm.sim.execution import simulate_fills, QuotePair  # type: ignore
    from crypto_mm.mm.inventory import Inventory  # type: ignore


# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------

def _apply_config_path_arg(cfg_path: Optional[str]) -> None:
    """Si fourni, force le chemin de config via variable d'env."""
    if cfg_path:
        os.environ["CRYPTO_MM_CONFIG"] = cfg_path


@contextmanager
def _squelch_console_logging():
    """
    Désactive globalement les logs (< CRITICAL) et redirige les StreamHandlers stdout -> stderr
    pour éviter de polluer la TTY pendant ws-live. Restaure l'état initial en sortie.
    """
    prev_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)

    modified: list[tuple[logging.Logger, logging.Handler, object]] = []
    try:
        all_names = [""] + [n for n in logging.root.manager.loggerDict.keys()]  # type: ignore[attr-defined]
        for name in all_names:
            lg = logging.getLogger(name)
            for h in list(getattr(lg, "handlers", [])):
                if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) in (sys.stdout, sys.__stdout__):
                    modified.append((lg, h, h.stream))
                    try:
                        h.flush()
                    except Exception:
                        pass
                    h.stream = sys.stderr
        yield
    finally:
        for lg, h, old_stream in modified:
            try:
                h.flush()
            except Exception:
                pass
            h.stream = old_stream
        logging.disable(prev_disable)


def _csv_appender(path: Path, header: Dict[str, str]):
    """
    Retourne une fonction append(rowdict) qui crée le fichier avec header si nécessaire.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    f = path.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=list(header.keys()))
    if created:
        w.writeheader()
        f.flush()

    def _append(row: Dict[str, object]) -> None:
        w.writerow(row)
        f.flush()
    return _append


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------

def build_parser(*, extras: bool = False) -> argparse.ArgumentParser:
    """Construit l'argparser. Les commandes extra ne sont ajoutées que si `extras=True`."""
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

    # Self-checks & WS
    sub.add_parser("check", help="Run quick checks (imports, clock, offline mapping).")
    sub.add_parser("ws-ping", help="Send a ping event to Kraken v2 and await first message.")

    p_tr = sub.add_parser("ws-trades", help="Stream Kraken trades (TTY refresh 500ms).")
    p_tr.add_argument("--pair", type=str, default="BTC/USD", help="Kraken pair (e.g., BTC/USD).")
    p_tr.add_argument("--duration", type=float, default=5.0, help="Duration in seconds.")

    p_ob = sub.add_parser("ws-book", help="Stream Kraken order book (snapshot + deltas + checksum).")
    p_ob.add_argument("--pair", type=str, default="BTC/USD", help="Kraken pair (e.g., BTC/USD).")
    p_ob.add_argument("--depth", type=int, default=10, help="Depth (10/25/100/500/1000).")
    p_ob.add_argument("--duration", type=float, default=5.0, help="Duration in seconds.")

    # Extras optionnels (ne doivent pas être présents pendant les tests)
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

        # --- E2E sim-live
        p_sim = sub.add_parser("sim-live", help="E2E: WS -> KPIs -> quotes -> risk -> exec -> inventory -> UI (+ CSV logs).")
        p_sim.add_argument("--symbol", type=str, default="BTC/USD")
        p_sim.add_argument("--depth", type=int, default=10)
        p_sim.add_argument("--min-size", type=float, default=0.01)
        p_sim.add_argument("--duration", type=float, default=120.0)

        p_sim.add_argument("--quote-size", type=float, default=0.25, help="Taille par côté (BTC)")
        p_sim.add_argument("--spread", type=float, default=10.0, help="Spread quote [$]")
        p_sim.add_argument("--k-skew", type=float, default=0.0, help="USD/BTC de skew par BTC d’inventaire")

        p_sim.add_argument("--kpi-window-sec", type=int, default=300)
        p_sim.add_argument("--latency-ms", type=int, default=50, help="Latence des quotes dans le simulateur")
        p_sim.add_argument("--priority", type=float, default=0.6, help="Probabilité d’être servi (0..1)")

        p_sim.add_argument("--logs-dir", type=str, default="logs")
        p_sim.add_argument("--fallback-replay", action="store_true", help="Si WS KO, bascule en mode replay synthétique")

    return parser


# --------------------------------------------------------------------------------------
# Démos utilitaires (sans WS)
# --------------------------------------------------------------------------------------

def cmd_live(settings: Settings, duration: float) -> int:
    """Boucle simple cadencée par le Clock (affiche latences p50/p95)."""
    clock = Clock(period_ms=settings.heartbeat_ms, logger=get_logger("live", settings.log_level))
    start = time.perf_counter()

    def on_tick(tick: int, dt_ms: float) -> None:
        elapsed = time.perf_counter() - start
        msg = (
            f"\r[{utc_now_iso()}] LIVE tick={tick} elapsed={elapsed:0.2f}s "
            f"dt={dt_ms:0.1f}ms p50={clock.p50_ms():0.1f}ms p95={clock.p95_ms():0.1f}ms"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()

    clock.run(duration_s=duration, iterations=None, on_tick=on_tick)
    sys.stdout.write("\n")
    return 0


def cmd_replay(settings: Settings, duration: float, speed: float) -> int:
    """Boucle “replay” plus rapide/lente via un facteur de vitesse."""
    clock = Clock(period_ms=int(settings.heartbeat_ms / max(speed, 0.1)), logger=get_logger("replay", settings.log_level))
    start = time.perf_counter()

    def on_tick(tick: int, dt_ms: float) -> None:
        elapsed = time.perf_counter() - start
        msg = (
            f"\r[{utc_now_iso()}] REPLAY tick={tick} elapsed={elapsed:0.2f}s "
            f"dt={dt_ms:0.1f}ms p50={clock.p50_ms():0.1f}ms p95={clock.p95_ms():0.1f}ms"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()

    clock.run(duration_s=duration, iterations=None, on_tick=on_tick)
    sys.stdout.write("\n")
    return 0


def cmd_plot(settings: Settings, output: Optional[str]) -> int:
    """Petit plot de démonstration (matplotlib)."""
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
    """Vérifications rapides (clock + checksum offline)."""
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


# --------------------------------------------------------------------------------------
# WS helpers
# --------------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------------
# WS: Live UI (extras)
# --------------------------------------------------------------------------------------

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
    Panneau droit  : Last trades (tsZ | type | side | qty | price) — issu de la TradeTape
    """
    from crypto_mm.data.trade_tape import TradeTape  # import local pour éviter cycles

    # Bloque toute écriture logging sur stdout pour préserver la TTY
    with _squelch_console_logging():
        client = KrakenWSV2Client(settings=settings, logger=get_logger("ws-live", settings.log_level))
        ob = OrderBookL2(logger=get_logger("ws-live.ob", settings.log_level))
        tape = TradeTape(capacity=2000, logger=get_logger("ws-live.tape", settings.log_level))

        state = LiveState(pair=pair, order_book=ob, trade_tape=tape, depth=depth, min_size=min_size)
        opts = RenderOptions(color=True)

        async def book_task() -> None:
            while True:
                try:
                    async for msg in client.subscribe_order_book(pair, depth=depth):
                        if msg.type == "snapshot":
                            ob.apply_snapshot(msg)
                        else:
                            ob.apply_delta(msg)
                except ResyncRequired:
                    # resync → repartir proprement
                    ob.clear()
                    continue

        async def trades_task() -> None:
            async for tr in client.subscribe_trades(pair):
                tape.push(tr)

        t1 = asyncio.create_task(book_task())
        t2 = asyncio.create_task(trades_task())

        start = time.perf_counter()
        try:
            while (time.perf_counter() - start) < duration:
                await asyncio.sleep(settings.heartbeat_ms / 1000.0)
                render(state, opts=opts)  # une seule fenêtre mise à jour in-place
        except KeyboardInterrupt:
            pass
        finally:
            stop_render()
            for t in (t1, t2):
                t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(t1, t2)

    return 0


# --------------------------------------------------------------------------------------
# SIM-LIVE (E2E): WS -> KPIs -> quotes -> risk -> exec -> inventory -> UI + CSV
# --------------------------------------------------------------------------------------
async def _sim_live_async(
    settings: Settings,
    symbol: str,
    duration: float,
    depth: int,
    min_size: float,
    quote_size: float,
    spread_usd: float,
    k_skew: float,
    kpi_window_sec: int,
    latency_ms: int,
    priority: float,
    logs_dir: str,
    fallback_replay: bool,
) -> int:
    # Pas de logs bruts dans la TTY (on désactive tout < CRITICAL et on reroute)
    with _squelch_console_logging():
        from .data.trade_tape import TradeTape
        from .ui.live_view import UiFill  # pour pousser les fills dans l'UI

        lg = get_logger("sim-live", settings.log_level)
        client = KrakenWSV2Client(settings=settings, logger=lg)
        ob = OrderBookL2(logger=get_logger("sim-live.ob", settings.log_level))
        tape = TradeTape(capacity=5000, logger=get_logger("sim-live.tape", settings.log_level))

        # UI
        state = LiveState(pair=symbol, order_book=ob, trade_tape=tape, depth=depth, min_size=min_size)
        opts = RenderOptions(color=True)

        # KPIs / Risk / Inventory
        kpis = KpiStore(window_s=kpi_window_sec)
        gate = RiskGate()
        inv = Inventory()

        # CSV loggers
        ld = Path(logs_dir)
        append_quotes = _csv_appender(ld / "quotes.csv", {
            "ts": "", "symbol": "", "action": "", "reason": "", "bid_px": "", "ask_px": "", "bid_qty": "", "ask_qty": "",
            "mid": "", "micro": "", "pos_btc": "", "pnl_r": "", "pnl_u": ""
        })
        append_trades = _csv_appender(ld / "trades.csv", {
            "ts": "", "symbol": "", "side": "", "px": "", "qty": "", "ts_trade": "", "ts_fill": ""
        })
        append_pnl = _csv_appender(ld / "pnl.csv", {
            "ts": "", "pos_btc": "", "avg_entry_px": "", "exposure_usd": "", "pnl_realized": "", "pnl_unrealized": ""
        })
        append_kpi = _csv_appender(ld / "kpi_spread.csv", {
            "ts": "", "size_btc": "", "spread_exec_usd": ""
        })

        # Collecteurs WS
        async def book_task() -> None:
            try:
                async for msg in client.subscribe_order_book(symbol, depth=depth):
                    if msg.type == "snapshot":
                        ob.apply_snapshot(msg)
                    else:
                        ob.apply_delta(msg)
            except Exception as e:
                lg.error({"event": "ws.book.error", "err": str(e)})
                if not fallback_replay:
                    raise

        async def trades_task() -> None:
            try:
                async for tr in client.subscribe_trades(symbol):
                    tape.push(tr)
            except Exception as e:
                lg.error({"event": "ws.trades.error", "err": str(e)})
                if not fallback_replay:
                    raise

        # Démarrage des tasks
        tasks: List[asyncio.Task] = []
        try:
            tasks = [asyncio.create_task(book_task()), asyncio.create_task(trades_task())]
        except Exception:
            if not fallback_replay:
                raise

        # Gestion Ctrl-C
        stop_flag = {"stop": False}
        def _on_sigint(*_): stop_flag["stop"] = True
        try:
            signal.signal(signal.SIGINT, _on_sigint)
        except Exception:
            pass

        start = time.perf_counter()
        last_trade_id = -1
        sizes_for_kpi = (0.1, 1.0, 5.0, 10.0)

        try:
            while not stop_flag["stop"] and (time.perf_counter() - start) < duration:
                await asyncio.sleep(settings.heartbeat_ms / 1000.0)
                now_epoch = time.time()

                # 1) KPIs de spread exécutable (book requis)
                top = ob.top(100)
                bids = top["bids"]; asks = top["asks"]
                if not bids or not asks:
                    continue  # pas de snapshot -> on retentera au heartbeat suivant

                for s_btc in sizes_for_kpi:
                    v = spread_for_size({"bids": bids, "asks": asks}, s_btc)
                    if v is not None:
                        kpis.append(now_epoch, s_btc, v)
                    append_kpi({
                        "ts": utc_now_iso(),
                        "size_btc": s_btc,
                        "spread_exec_usd": "" if v is None else f"{v:.4f}",
                    })

                # 2) Fair price (micro de préférence, sinon mid)
                fair = ob.microprice()
                if not math.isfinite(fair):
                    fair = ob.mid_price()
                if not math.isfinite(fair):
                    continue  # rien à faire tant qu'on n'a pas de prix "fair"

                # Valeurs mid/micro pour logs/UI
                mid = ob.mid_price()
                micro = ob.microprice()

                # 3) Quotes
                q = generate_quote(
                    fair_price=float(fair),
                    spread_quote=float(spread_usd),
                    quote_size=float(quote_size),
                    tick=0.1,
                    inventory=float(inv.state().get("position_btc", 0.0) or 0.0),
                    k_skew=float(k_skew),
                    min_quote_size=0.0,
                )

                # 4) Risk Gate
                inv_state = inv.state()
                pnl_state = {
                    "pnl_realized": inv_state.get("pnl_realized", 0.0) or 0.0,
                    "pnl_unrealized": inv_state.get("pnl_unrealized", 0.0) or 0.0,
                }
                dec: Decision = gate.evaluate(
                    quotes=type("Q", (), {"bid_px": q.bid_px, "ask_px": q.ask_px, "bid_qty": q.bid_sz, "ask_qty": q.ask_sz}),
                    inv_state={"position_btc": inv_state.get("position_btc", 0.0) or 0.0},
                    pnl_state=pnl_state,
                    mid=float(mid if math.isfinite(mid) else fair),
                )

                # 5) Exécution simulée (nouveaux trades depuis dernier heartbeat)
                trades_new = []
                for tr in tape.last(1000):
                    try:
                        tid = int(tr.trade_id)
                    except Exception:
                        continue
                    if tid > last_trade_id:
                        trades_new.append(tr)
                if trades_new:
                    trades_new.sort(key=lambda t: int(t.trade_id))
                    last_trade_id = int(trades_new[-1].trade_id)

                fills = []
                if dec.action in ("allow", "adjust") and trades_new:
                    bid_px = dec.bid_px if dec.bid_px is not None else q.bid_px
                    ask_px = dec.ask_px if dec.ask_px is not None else q.ask_px
                    bid_qty = dec.bid_qty if dec.bid_qty is not None else q.bid_sz
                    ask_qty = dec.ask_qty if dec.ask_qty is not None else q.ask_sz

                    qp = QuotePair(
                        bid_px=float(bid_px), ask_px=float(ask_px),
                        bid_qty=float(bid_qty), ask_qty=float(ask_qty),
                        ts_place_epoch=now_epoch - (settings.heartbeat_ms / 1000.0),  # place un HB avant
                    )
                    fills = simulate_fills(
                        trades=trades_new,
                        quotes=qp,
                        latency_ms=int(latency_ms),
                        p_priority_bid=float(priority),
                        p_priority_ask=float(priority),
                        rng=None,
                    )
                    for f in fills:
                        inv.on_fill(f)  # <— ta signature: on_fill(self, fill: FillEvent)
                        append_trades({
                            "ts": utc_now_iso(), "symbol": symbol, "side": f.side,
                            "px": f"{f.px:.2f}", "qty": f"{f.qty:.6f}",
                            "ts_trade": f"{f.ts_trade:.6f}", "ts_fill": f"{f.ts_fill:.6f}",
                        })
                        # Alimente l'UI STRATEGY (derniers fills)
                        state.fills.append(UiFill(ts_epoch=f.ts_fill, side=f.side, px=f.px, qty=f.qty))

                # 6) Mark-to-market
                inv.mark(mid_px=float(mid if math.isfinite(mid) else fair))
                st = inv.state()
                append_pnl({
                    "ts": utc_now_iso(),
                    "pos_btc": f"{st.get('position_btc', 0.0):.6f}",
                    "avg_entry_px": "" if st.get('avg_entry_px') is None else f"{st['avg_entry_px']:.6f}",
                    "exposure_usd": f"{(st.get('exposure_usd', 0.0) or 0.0):.2f}",
                    "pnl_realized": f"{(st.get('pnl_realized', 0.0) or 0.0):.2f}",
                    "pnl_unrealized": f"{(st.get('pnl_unrealized', 0.0) or 0.0):.2f}",
                })

                # Met à jour l'état STRATEGY pour l'UI
                state.pnl_realized = (st.get("pnl_realized", 0.0) or 0.0)
                state.pnl_unrealized = (st.get("pnl_unrealized", 0.0) or 0.0)
                state.position_btc = (st.get("position_btc", 0.0) or 0.0)
                state.avg_entry_px = st.get("avg_entry_px", None)
                state.exposure_usd = (st.get("exposure_usd", 0.0) or 0.0)
                state.inv_history.append(state.position_btc)

                # 7) Log des quotes envoyées au simulateur (CSV uniquement)
                append_quotes({
                    "ts": utc_now_iso(),
                    "symbol": symbol,
                    "action": dec.action,
                    "reason": dec.reason,
                    "bid_px": f"{(dec.bid_px if dec.bid_px is not None else q.bid_px):.2f}",
                    "ask_px": f"{(dec.ask_px if dec.ask_px is not None else q.ask_px):.2f}",
                    "bid_qty": f"{(dec.bid_qty if dec.bid_qty is not None else q.bid_sz):.6f}",
                    "ask_qty": f"{(dec.ask_qty if dec.ask_qty is not None else q.ask_sz):.6f}",
                    "mid": f"{mid:.2f}" if math.isfinite(mid) else "",
                    "micro": f"{micro:.2f}" if math.isfinite(micro) else "",
                    "pos_btc": f"{state.position_btc:.6f}",
                    "pnl_r": f"{state.pnl_realized:.2f}",
                    "pnl_u": f"{state.pnl_unrealized:.2f}",
                })

                # 8) UI refresh
                render(state, opts=opts)

        except KeyboardInterrupt:
            pass
        finally:
            stop_render()
            for t in tasks:
                t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                if tasks:
                    await asyncio.gather(*tasks)

        return 0



def cmd_ws_ping(settings: Settings) -> int:
    return asyncio.run(_ws_ping_async(settings))


def cmd_ws_trades(settings: Settings, pair: str, duration: float) -> int:
    return asyncio.run(_ws_trades_async(settings, pair, duration))


def cmd_ws_book(settings: Settings, pair: str, duration: float, depth: int) -> int:
    return asyncio.run(_ws_book_async(settings, pair, duration, depth))


def cmd_ws_live(settings: Settings, pair: str, duration: float, depth: int, min_size: float) -> int:
    return asyncio.run(_ws_live_async(settings, pair, duration, depth, min_size))


def cmd_sim_live(
    settings: Settings,
    symbol: str,
    duration: float,
    depth: int,
    min_size: float,
    quote_size: float,
    spread_usd: float,
    k_skew: float,
    kpi_window_sec: int,
    latency_ms: int,
    priority: float,
    logs_dir: str,
    fallback_replay: bool,
) -> int:
    return asyncio.run(
        _sim_live_async(
            settings, symbol, duration, depth, min_size,
            quote_size, spread_usd, k_skew, kpi_window_sec,
            latency_ms, priority, logs_dir, fallback_replay
        )
    )


# --------------------------------------------------------------------------------------
# Entrée CLI
# --------------------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    # Extras activables uniquement via ENV pour l'exécutable,
    # mais PAS pris en compte par tests qui appellent build_parser() sans arg.
    extras_env = os.environ.get("CRYPTO_MM_EXTRAS", "0").lower() in {"1", "true", "yes", "on"}
    parser = build_parser(extras=extras_env)

    args = parser.parse_args(argv)
    if args.config:
        _apply_config_path_arg(args.config)

    settings = load_settings()

    if args.cmd == "live":
        return cmd_live(settings, duration=args.duration)
    if args.cmd == "replay":
        return cmd_replay(settings, duration=args.duration, speed=args.speed)
    if args.cmd == "plot":
        return cmd_plot(settings, output=args.output)
    if args.cmd == "check":
        return cmd_check(settings)
    if args.cmd == "ws-ping":
        return cmd_ws_ping(settings)
    if args.cmd == "ws-trades":
        return cmd_ws_trades(settings, pair=args.pair, duration=args.duration)
    if args.cmd == "ws-book":
        return cmd_ws_book(settings, pair=args.pair, duration=args.duration, depth=args.depth)

    # Extras
    if extras_env and args.cmd == "ws-book-ladder":
        # Cette commande doit exister dans votre codebase si vous l'utilisez.
        from .ui.main import cmd_ws_book_ladder  # import tardif pour éviter dépendances pendant les tests
        return cmd_ws_book_ladder(settings, pair=args.pair, duration=args.duration, depth=args.depth, min_size=args.min_size)
    if extras_env and args.cmd == "ws-live":
        return cmd_ws_live(settings, pair=args.pair, duration=args.duration, depth=args.depth, min_size=args.min_size)
    if extras_env and args.cmd == "sim-live":
        return cmd_sim_live(
            settings,
            symbol=args.symbol,
            duration=args.duration,
            depth=args.depth,
            min_size=args.min_size,
            quote_size=args.quote_size,
            spread_usd=args.spread,
            k_skew=args.k_skew,
            kpi_window_sec=args.kpi_window_sec,
            latency_ms=args.latency_ms,
            priority=args.priority,
            logs_dir=args.logs_dir,
            fallback_replay=args.fallback_replay,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

