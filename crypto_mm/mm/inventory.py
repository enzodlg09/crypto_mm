from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from ..sim.execution import FillEvent


@dataclass
class Inventory:
    """
    Gestion d'inventaire & PnL (WAP, realized/unrealized).

    État tenu en continu (toutes les unités sont float):
    - position_btc : >0 long, <0 short.
    - avg_entry_px : WAP (USD/BTC) de la position ouverte, None si flat.
    - pnl_realized : USD réalisés sur les fermetures partielles/totales.
    - last_mid     : dernier mid de valorisation (optionnel).

    Notes
    -----
    - Pour une position short, `avg_entry_px` représente le prix moyen de vente
      (entrée short). Le calcul de PnL réalisé/unréalisé s'adapte au signe.
    """

    position_btc: float = 0.0
    avg_entry_px: Optional[float] = None
    pnl_realized: float = 0.0
    last_mid: Optional[float] = None

    # ---- API ---------------------------------------------------------------

    def on_fill(self, fill: FillEvent) -> None:
        """
        Intègre un fill dans l'état (WAP incrémental + PnL réalisé).

        Parameters
        ----------
        fill : FillEvent
            Exécution buy/sell au prix `fill.px` et quantité `fill.qty`.
        """
        side = fill.side
        px = float(fill.px)
        qty = float(fill.qty)
        q = self.position_btc
        w = self.avg_entry_px

        if side == "buy":
            if q < 0:  # on couvre un short (partiel ou total)
                closing = min(qty, -q)
                self.pnl_realized += (w - px) * closing  # short: vendu w, racheté px
                q += closing  # q augmente (vers 0)
                qty -= closing
                if q == 0.0 and qty == 0.0:
                    self.position_btc = 0.0
                    self.avg_entry_px = None
                    return
                if q == 0.0 and qty > 0.0:
                    # on bascule long sur le reliquat
                    self.position_btc = qty
                    self.avg_entry_px = px
                    return
                # sinon short partiellement couvert, WAP inchangé
                self.position_btc = q
                # qty==0 ici
                return
            else:
                # on augmente (ou crée) un long
                new_q = q + qty
                new_w = px if q == 0.0 else (w * q + px * qty) / new_q
                self.position_btc = new_q
                self.avg_entry_px = new_w
                return

        elif side == "sell":
            if q > 0:  # on déboucle un long
                closing = min(qty, q)
                self.pnl_realized += (px - w) * closing  # long: acheté w, vendu px
                q -= closing
                qty -= closing
                if q == 0.0 and qty == 0.0:
                    self.position_btc = 0.0
                    self.avg_entry_px = None
                    return
                if q == 0.0 and qty > 0.0:
                    # bascule short pour le reliquat
                    self.position_btc = -qty
                    self.avg_entry_px = px
                    return
                # long partiellement débouclé, WAP inchangé
                self.position_btc = q
                return
            else:
                # on augmente (ou crée) un short
                new_q = q - qty
                mag_old = -q
                mag_new = -new_q
                new_w = px if mag_old == 0.0 else (self.avg_entry_px * mag_old + px * qty) / mag_new
                self.position_btc = new_q
                self.avg_entry_px = new_w
                return
        else:
            raise ValueError(f"Unknown side {side!r}")

    def mark(self, mid_px: float) -> None:
        """Met à jour le dernier mid pour valorisation."""
        self.last_mid = float(mid_px)

    # ---- propriétés / sérialisation ---------------------------------------

    @property
    def exposure_usd(self) -> Optional[float]:
        """Exposition (USD) = position_btc * last_mid (si disponible)."""
        if self.last_mid is None:
            return None
        return self.position_btc * self.last_mid

    @property
    def pnl_unrealized(self) -> Optional[float]:
        """PnL non réalisé basé sur `last_mid` et `avg_entry_px`."""
        if self.avg_entry_px is None or self.last_mid is None or self.position_btc == 0.0:
            return None if self.last_mid is None else 0.0
        # même formule pour long/short : (mid - wap) * q
        return (self.last_mid - self.avg_entry_px) * self.position_btc

    @property
    def pnl_total(self) -> Optional[float]:
        """PnL total (réalisé + non réalisé si valorisé)."""
        ur = self.pnl_unrealized
        if ur is None:
            return None
        return self.pnl_realized + ur

    def state(self) -> Dict[str, Optional[float]]:
        """Retourne un dict sérialisable de l'état courant."""
        return {
            "position_btc": self.position_btc,
            "avg_entry_px": self.avg_entry_px,
            "exposure_usd": self.exposure_usd,
            "pnl_realized": self.pnl_realized,
            "pnl_unrealized": self.pnl_unrealized,
            "pnl_total": self.pnl_total,
            "last_mid": self.last_mid,
        }
