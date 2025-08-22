from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Q:
    bid_px: float
    ask_px: float
    bid_qty: float
    ask_qty: float


@dataclass(frozen=True)
class Decision:
    action: str                   # 'allow' | 'adjust' | 'block'
    reason: str
    bid_px: Optional[float] = None
    ask_px: Optional[float] = None
    bid_qty: Optional[float] = None
    ask_qty: Optional[float] = None


@dataclass
class RiskGate:
    # Hard limits
    max_notional_usd: float = 1_000_000.0
    max_loss_usd: float = 100_000.0
    hysteresis_usd: float = 0.0          # marge absolue pour relâcher le hard loss

    # Soft limits
    inventory_cap_btc: float = 10.0
    drawdown_trigger_usd: float = 500.0
    cooldown_hb: int = 2

    # Ajustements “soft”
    soft_center_shift_per_excess_btc: float = 0.0  # USD/BTC de shift du centre par BTC d’excès
    soft_min_scale: float = 0.2                    # plancher de réduction des tailles

    # État interne
    _hb: int = 0
    _cooldown_until: Optional[int] = None
    _cooldown_reason: Optional[str] = None
    _pnl_peak: float = 0.0
    _blocked_due_to_loss: bool = False
    _drawdown_armed: bool = True  # edge-trigger : ré-armé uniquement si nouveau peak atteint

    # ---------------------------------------------------------------------

    def _start_cooldown(self, reason: str) -> None:
        self._cooldown_until = self._hb + max(0, int(self.cooldown_hb))
        self._cooldown_reason = reason

    def evaluate(self, quotes: Q, inv_state: dict, pnl_state: dict, mid: float) -> Decision:
        self._hb += 1

        # --- PnL courant & peak
        pnl_realized = float(pnl_state.get("pnl_realized", 0.0))
        pnl_unrealized = float(pnl_state.get("pnl_unrealized", 0.0))
        pnl_total = pnl_realized + pnl_unrealized

        # réarmement si nouveau peak
        if pnl_total > self._pnl_peak + 1e-9:
            self._pnl_peak = pnl_total
            self._drawdown_armed = True

        # --- Hard: notional
        pos_btc = float(inv_state.get("position_btc", 0.0))
        notional = abs(pos_btc * mid)
        if notional > self.max_notional_usd + 1e-9:
            # Préfixe conforme au test
            self._start_cooldown("hard.notional")
            return Decision(action="block", reason="hard.notional")

        # --- Hard: max loss avec histérèse absolue
        if pnl_total < -self.max_loss_usd - 1e-9:
            self._blocked_due_to_loss = True
        if self._blocked_due_to_loss:
            # reste bloqué tant que total < -(max_loss_usd - hysteresis_usd)
            release_level = -(self.max_loss_usd - max(0.0, self.hysteresis_usd))
            if pnl_total < release_level - 1e-9:
                return Decision(action="block", reason="hard.loss")
            else:
                self._blocked_due_to_loss = False  # relâchement

        # --- Cooldown (quel qu’en soit le motif), si en cours
        if self._cooldown_until is not None and self._hb < self._cooldown_until:
            return Decision(action="block", reason=self._cooldown_reason or "cooldown")
        else:
            self._cooldown_until = None
            self._cooldown_reason = None

        # --- Soft: drawdown edge-trigger
        dd = max(0.0, self._pnl_peak - pnl_total)
        if self._drawdown_armed and dd > self.drawdown_trigger_usd + 1e-9 and pnl_total < 0.0:
            self._start_cooldown("soft.drawdown")
            self._drawdown_armed = False  # ne re-déclenche pas tant qu’on n’a pas fait un nouveau peak
            return Decision(action="block", reason="soft.drawdown")

        # --- Soft: inventaire -> ajustements (pas de block)
        cap = max(1e-12, float(self.inventory_cap_btc))
        abs_pos = abs(pos_btc)

        if abs_pos <= cap + 1e-12:
            # Dans le cap : éventuellement un léger shrink “proche du cap”, sinon allow tel quel
            return Decision(
                action="allow",
                reason="ok",
                bid_px=quotes.bid_px, ask_px=quotes.ask_px,
                bid_qty=quotes.bid_qty, ask_qty=quotes.ask_qty,
            )

        # Au-delà du cap : “adjust”
        excess = abs_pos - cap

        # Shift du centre (sens : long -> centre ↓ ; short -> centre ↑)
        center = 0.5 * (quotes.bid_px + quotes.ask_px)
        half = 0.5 * (quotes.ask_px - quotes.bid_px)
        shift = self.soft_center_shift_per_excess_btc * excess
        if pos_btc > 0:
            center_adj = center - shift  # long -> vers le bas
        else:
            center_adj = center + shift  # short -> vers le haut
        bid_px_adj = center_adj - half
        ask_px_adj = center_adj + half

        # Réduction des tailles (plancher soft_min_scale)
        scale = max(self.soft_min_scale, min(1.0, cap / abs_pos))
        bid_qty_adj = quotes.bid_qty * scale
        ask_qty_adj = quotes.ask_qty * scale

        return Decision(
            action="adjust",
            reason="soft.inventory",
            bid_px=bid_px_adj, ask_px=ask_px_adj,
            bid_qty=bid_qty_adj, ask_qty=ask_qty_adj,
        )
