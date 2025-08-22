from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Optional, Tuple, Union
from datetime import datetime, timezone

from ..mm.quoting import generate_quote
from ..mm.inventory import Inventory


Number = Union[int, float]


@dataclass(frozen=True)
class ReplayMetrics:
    fills: int
    pnl_realized: float
    pnl_unrealized: float
    pnl_total: float
    avg_cycle_ms: float


def _parse_ts(ts: Union[str, Number]) -> float:
    """Accepte epoch (float/int) ou ISO8601."""
    if isinstance(ts, (int, float)):
        return float(ts)
    # ISO → epoch
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return dt.timestamp()


class _Event:
    __slots__ = ("type", "ts", "payload")

    def __init__(self, typ: str, ts: float, payload: dict):
        self.type = typ
        self.ts = ts
        self.payload = payload


class ReplayRunner:
    """
    Rejoue un fichier JSONL (événements ordonnés par 'ts') et simule une stratégie simple :
      - À CHAQUE ÉVÉNEMENT, on (re)génère une quote (bid/ask) autour du 'fair'
      - On tente des fills déterministes contre l'événement 'trade' courant
      - Pas d'attente réelle : la vitesse 'speed' est uniquement utilisée pour calculer avg_cycle_ms
    Le but est 100% déterministe pour les tests d’intégration.
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        *,
        speed: float = 1.0,
        tick: float = 0.1,
        quote_size: float = 0.5,
        spread_usd: float = 1.0,
        k_skew: float = 0.0,
        min_quote_size: float = 0.0,
        seed_mid: Optional[float] = None,
    ) -> None:
        self.path = Path(jsonl_path)
        self.speed = float(speed if speed > 0 else 1.0)
        self.tick = float(tick)
        self.quote_size = float(quote_size)
        self.spread_usd = float(spread_usd)
        self.k_skew = float(k_skew)
        self.min_quote_size = float(min_quote_size)
        self.seed_mid = seed_mid

        self._events: List[_Event] = []

    # ------------------------------- I/O -------------------------------

    def _load(self) -> None:
        evts: List[_Event] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                typ = str(obj.get("type", ""))
                ts = _parse_ts(obj.get("ts"))
                payload = {k: v for k, v in obj.items() if k not in {"type", "ts"}}
                evts.append(_Event(typ, ts, payload))
        evts.sort(key=lambda e: e.ts)
        self._events = evts

    # --------------------------- core logic ---------------------------

    @staticmethod
    def _snap_price(px: float, tick: float) -> float:
        if tick <= 0:
            return px
        # snap half-up
        return round(px / tick) * tick

    def _deterministic_fill(
        self,
        *,
        trade_price: float,
        trade_qty: float,
        bid_px: float,
        ask_px: float,
        bid_sz_rem: float,
        ask_sz_rem: float,
    ) -> List[Tuple[str, float, float]]:
        """
        Renvoie une liste de fills (side, px, qty) contre un trade unique.
        Règle simple déterministe :
          - trade_price <= bid_px  -> on achète (buy) jusqu'au restant bid
          - trade_price >= ask_px  -> on vend (sell) jusqu'au restant ask
          - sinon aucun fill
        """
        out: List[Tuple[str, float, float]] = []
        if trade_price <= bid_px and bid_sz_rem > 0:
            qty = min(bid_sz_rem, trade_qty)
            if qty > 0:
                out.append(("buy", bid_px, qty))
        elif trade_price >= ask_px and ask_sz_rem > 0:
            qty = min(ask_sz_rem, trade_qty)
            if qty > 0:
                out.append(("sell", ask_px, qty))
        return out

    def run(self) -> ReplayMetrics:
        """
        Exécute le replay de manière purement déterministe.
        - Ne dépend PAS de l'horloge murale
        - 'avg_cycle_ms' est calculé depuis les timestamps des événements, ajustés par 'speed'
        """
        self._load()
        if not self._events:
            return ReplayMetrics(0, 0.0, 0.0, 0.0, 0.0)

        inv = Inventory()
        last_ts = None
        last_trade_price: Optional[float] = None
        fills_count = 0
        cycle_deltas_ms: List[float] = []

        # Si premiers deltas 'book' sans snapshot, on exige un seed mid
        seen_snapshot = any(e.type == "snapshot" for e in self._events[:1])
        fair = float(self.seed_mid) if self.seed_mid is not None else None
        if fair is None:
            # si pas de seed et pas de snapshot au début, on tolère mais le premier trade fixera 'fair'
            pass

        for e in self._events:
            if last_ts is not None:
                dt_ms = (e.ts - last_ts) * 1000.0 / self.speed
                cycle_deltas_ms.append(dt_ms)
            last_ts = e.ts

            # 1) Met à jour le fair selon l’événement précédent connu
            #    Priorité : snapshot.mid -> book.mid -> dernier trade -> seed_mid
            if e.type == "snapshot":
                # On accepte plusieurs formats: {"mid":...} ou {"best_bid":..., "best_ask":...}
                if "mid" in e.payload:
                    fair = float(e.payload["mid"])
                elif "best_bid" in e.payload and "best_ask" in e.payload:
                    bb = float(e.payload["best_bid"])
                    aa = float(e.payload["best_ask"])
                    fair = (bb + aa) / 2.0
            elif e.type == "book":
                if "mid" in e.payload:
                    fair = float(e.payload["mid"])
                elif "best_bid" in e.payload and "best_ask" in e.payload:
                    bb = float(e.payload["best_bid"])
                    aa = float(e.payload["best_ask"])
                    fair = (bb + aa) / 2.0

            # Si on n'a toujours pas de fair, on peut l’amorcer au dernier trade connu
            if fair is None and last_trade_price is not None:
                fair = float(last_trade_price)
            if fair is None and self.seed_mid is not None:
                fair = float(self.seed_mid)

            # 2) Génère la quote pour CE cycle (même si l'événement n'est pas un trade)
            #    NB: l'inventaire (skew) influence légèrement les prix si k_skew != 0
            inv_state = inv.state()
            q = generate_quote(
                fair_price=float(fair if fair is not None else (last_trade_price if last_trade_price is not None else 0.0)),
                spread_quote=self.spread_usd,
                quote_size=self.quote_size,
                tick=self.tick,
                inventory=float(inv_state.get("position_btc", 0.0) or 0.0),
                k_skew=self.k_skew,
                min_quote_size=self.min_quote_size,
            )

            # 3) Si trade, simule le(s) fill(s) déterministe(s) pour ce cycle
            if e.type == "trade":
                trade_price = float(e.payload["price"])
                trade_qty = float(e.payload["qty"])
                last_trade_price = trade_price

                bid_rem = q.bid_sz
                ask_rem = q.ask_sz

                fl = self._deterministic_fill(
                    trade_price=trade_price,
                    trade_qty=trade_qty,
                    bid_px=q.bid_px,
                    ask_px=q.ask_px,
                    bid_sz_rem=bid_rem,
                    ask_sz_rem=ask_rem,
                )
                for side, px, qty in fl:
                    # Inventory.on_fill(fill: FillEvent-like)
                    inv.on_fill(SimpleNamespace(side=side, px=px, qty=qty))
                fills_count += len(fl)

            # 4) Mark-to-market à chaque cycle (fair → mid)
            if fair is not None:
                inv.mark(mid_px=float(fair))

        # métriques finales
        st = inv.state()
        pnl_r = float(st.get("pnl_realized", 0.0) or 0.0)
        pnl_u = float(st.get("pnl_unrealized", 0.0) or 0.0)
        pnl_t = float((st.get("pnl_total", pnl_r + pnl_u)) or (pnl_r + pnl_u))
        avg_ms = (sum(cycle_deltas_ms) / len(cycle_deltas_ms)) if cycle_deltas_ms else 0.0

        return ReplayMetrics(
            fills=fills_count,
            pnl_realized=pnl_r,
            pnl_unrealized=pnl_u,
            pnl_total=pnl_t,
            avg_cycle_ms=avg_ms,
        )
