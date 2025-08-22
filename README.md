# README

# crypto\_mm — Market Making (Kraken WS v2)

Dans l'example.env, Il y a la clef d'api si vous voulez run. Elle est active jusqu'au 20 septembre.

### 1) Installation (Windows PowerShell)

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
# (optionnel, pour export Parquet)
pip install "pyarrow>=14"
```

### 1) Installation (macOS / Linux / WSL)

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
# optionnel Parquet
pip install "pyarrow>=14"
```

### 2) Activer les “extras” (UI live, sim-live)

* PowerShell :

```powershell
$env:CRYPTO_MM_EXTRAS="1"
```

* macOS/Linux/WSL :

```bash
export CRYPTO_MM_EXTRAS=1
```

### 3) Lancer l’UI “live order book + trades”

```bash
python -m crypto_mm.main ws-live --pair BTC/USD --depth 25 --min-size 0.005 --duration 120
```

### 4) Lancer la simulation E2E (live stream → KPIs → quoting → risk → fills simulés → inventaire → UI + CSV)

> Paramètres un peu agressifs pour “voir bouger” l’inventaire.

```bash
python -m crypto_mm.main sim-live \
  --symbol BTC/USD \
  --depth 25 \
  --min-size 0.005 \
  --duration 300 \
  --quote-size 0.75 \
  --spread 3 \
  --k-skew 0.80 \
  --kpi-window-sec 120 \
  --latency-ms 8 \
  --priority 0.94 \
  --logs-dir logs
```

> Les CSV sont écrits dans `logs/` : `quotes.csv`, `trades.csv`, `pnl.csv`, `kpi_spread.csv` (rotation par jour si vous utilisez les *sinks*).

---

## Architecture (ASCII)

```
                       ┌────────────────────────────────────────────────────────┐
  Kraken WS v2         │                    crypto_mm                           │
 (trade + book)        └────────────────────────────────────────────────────────┘
      │                                │
      ├──────────────► OrderBookL2 ◄───┤       (stateful, mid/microprice, ladder)
      │                                │
      └──────────────► TradeTape  ◄────┘       (buffer des derniers trades)

      ┌────────────────────────────────────────────────────────────────────────┐
      │  KPIs (spread exécutable par taille)   →   KpiStore(window)            │
      │  Quoting (fair ± half, skew inventaire, snapping, anti-cross)         │
      │  RiskGate (allow/adjust/block)                                        │
      │  Simulation d’exécution (latence, priorité)  →  Fills → Inventory     │
      │  Mark-to-market → PnL R/U/T                                           │
      └────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ├───► UI (TTY)  [book + KPIs + ladder + stratégie + trades]
                                       └───► Logs CSV/Parquet (quotes, trades, pnl, kpis)
```

---

## Live vs Replay

* **Live** : consomme directement le WS Kraken v2.

  * `ws-live`: UI simple (book+trades).
  * `sim-live`: pipeline complet + simulation d’exécution + inventaire + PnL.

* **Replay** : rejoue un dump **JSONL** (déterministe) avec `sim/replay.py`.

  * Exemple :

    ```bash
    python -m crypto_mm.sim.replay \
      --file examples/replay/sample.jsonl \
      --speed 10 \
      --tick 0.1 \
      --quote-size 0.25 \
      --spread-usd 1.0 \
      --k-skew 0.05 \
      --min-quote-size 0.0
    ```
  * Données : lignes `{"type":"book"|"trade", ...}` ordonnées par `ts`.
    Voir `examples/replay/sample.jsonl` ci-dessous.

---

## Paramètres clefs (MM)

* `--quote-size` : taille par côté en BTC (ex: `0.25`, `0.75`).
* `--spread` : spread total en USD (ex: `3` → ±1.5 autour du fair).
* `--k-skew` : coefficient de skew (USD/BTC par BTC d’inventaire).
  Long → bid ↑, ask ↓ (et réciproquement).
* `--latency-ms` : latence appliquée à vos quotes dans le simulateur (impacte les fills).
* `--priority` : probabilité (\~0..1) que votre quote passe en priorité (proxy d’alpha/queue).
* `--kpi-window-sec` : fenêtre de calcul des médianes (exéc. spreads).
* `--depth` / `--min-size` : profondeur et filtre des niveaux affichés.
* `--logs-dir` : dossier de sortie CSV (quotes/trades/pnl/kpi).

---

## Limites de risque

La classe `RiskGate` renvoie une décision **`allow` / `adjust` / `block`** en fonction de :

* la proposition de quote (prix/sizes),
* l’inventaire courant (BTC),
* PnL réalisé & non-réalisé,
* et le mid de marché (pour la valorisation).

Les heuristiques sont codées dans `risk/limits.py` (faciles à modifier).
Dans la simulation, une décision `block` annule l’envoi de quotes ; `adjust` peut réduire les tailles/prix.

---

## UI – aperçu (TTY)

```
BTC/USD  @ 2025-08-22T12:27:59Z
================================================================================================
 ORDER BOOK — TOP 10 LEVELS (cumulative sizes)                                 │ tsZ                  type  side         qty        price
 BID px  cum_btc  ||  ASK px  cum_btc                                          │ 2025-08-22T12:27:50Z trade sell     0.002260   112,011.10
 112,011.10  4.4202 || 112,011.20  1.8831                                      │ 2025-08-22T12:27:51Z trade buy      0.000717   112,011.20
 ...                                                                            │ ...
 Spread: $0.10 (1 ticks) | Mid: 112,011.15 | Microprice: 112,011.17
--------------------------------------------------------------------------------
 EXEC SPREADS (instant)  0.1: $0.10 | 1: $0.10 | 5: $0.90 | 10: $4.18
 EXEC SPREADS (5m median) 0.1: $0.10 | 1: $0.10 | 5: $1.01 | 10: $5.28
 SPREAD KPI (median only) — size | median | n_obs
   0.1 |   0.10 |    162
     1 |   0.10 |    162
     5 |   1.01 |    162
    10 |   5.28 |    159
 MICRO VOL (last 50) — stdev: $12.34  |  min/max: 111,995.2 .. 112,024.8

 LADDER (min_size=0.005, depth=25)
 BIDS ≥0.005: lvls=19 sum=11.7151 BTC | depth≈ $1.31M | range 112184..112253
 ASKS ≥0.005: lvls=23 sum=26.0805 BTC | depth≈ $2.93M | range 112237..112270

 STRATEGY — PnL / Position / Inventory / Sim Fills
 PnL R/U/T: +0.68 / -668.05 / -667.37  USD
 POS: -3.512537 BTC  |  AvgPx: 112,257.06  |  Exposure: -394,975.12 USD
 INV: last=-3.512537 BTC  [-3.512537 .. -2.5391]  ████▇▇▇▇▁▁▁...
 SIM FILLS (last)
 2025-08-22T12:47:55Z  sell  qty=0.000867  px=112,257.80
 2025-08-22T12:47:58Z  sell  qty=0.013582  px=112,258.60
 ...
```

> Notes :
>
> * La **volatilité micro** est calculée sur une fenêtre roulante de **50** micro-prix.
> * Le **spread** affiché est clampé à `≥ 0` (en cas d’inversions transitoires).
> * Les colonnes de la **KPI table** sont **alignées** et limitées pour rester lisibles.

---

## Export & Persistance

Par défaut, la simulation écrit **CSV** dans `--logs-dir` :

* `quotes.csv` (quotes envoyées & décisions risk),
* `trades.csv` (fills simulés, avec `ts_trade` et `ts_fill`),
* `pnl.csv` (position, avg\_entry, exposure, PnL R/U),
* `kpi_spread.csv` (mesures de spread exécutable par taille).

Si vous installez `pyarrow`, vous pouvez activer des *sinks* Parquet (rotation quotidienne, écriture atomique).



## Exemples de configs (à créer)

### `examples/config.yaml`

```yaml
# Heartbeat UI (ms)
heartbeat_ms: 500

# Logging minimal (UI propre)
log_level: WARNING

# Kraken WS v2
ws_url_public: wss://ws.kraken.com/v2

# Répertoire de plots (si vous utilisez --plot)
plot_output_dir: ./plots
```

### `examples/replay/sample.jsonl`

```json
{"type":"book","ts":10.0,"best_bid":99.9,"best_ask":100.1}
{"type":"trade","ts":11.0,"price":100.60,"qty":0.25}
{"type":"trade","ts":12.0,"price":100.40,"qty":0.10}
```

> Utilisez ce fichier pour vérifier la reproductibilité du mode replay.

### Merci & bon run ✨

---

# 1) Self-checks / utilitaires

**Ping WS Kraken v2**

```bash
python -m crypto_mm.main ws-ping
```

**Vérif rapide (clock + checksum offline)**

```bash
python -m crypto_mm.main check
```

**Petit plot de démo (matplotlib)**

```bash
python -m crypto_mm.main plot --output plots/demo.png
```

---

# 2) Streams simples (sans UI avancée)

**Trades stream (console rolling) – 60s**

```bash
python -m crypto_mm.main ws-trades --pair BTC/USD --duration 60
```

**Order book L2 (seq + meilleures quotes) – 60s**

```bash
python -m crypto_mm.main ws-book --pair BTC/USD --depth 25 --duration 60
```

---

# 3) UI Live (OrderBook + Trades)

**Vue TTY live (book + trades + KPIs de base)**

```bash
python -m crypto_mm.main ws-live --pair BTC/USD --depth 25 --min-size 0.005 --duration 120
```

> Astuce largeur terminal : maximise ta fenêtre pour voir toutes les colonnes.

---

# 4) SIM-LIVE (E2E) — WS → KPIs → Quoting → Risk → Fills simulés → Inventaire → UI + CSV

## 4.1) Profil “conservateur” (peu de fills)

```bash
python -m crypto_mm.main sim-live \
  --symbol BTC/USD \
  --depth 25 \
  --min-size 0.01 \
  --duration 180 \
  --quote-size 0.25 \
  --spread 6 \
  --k-skew 0.20 \
  --kpi-window-sec 300 \
  --latency-ms 50 \
  --priority 0.6 \
  --logs-dir logs
```

## 4.2) Profil “agressif” (fills fréquents, inventaire bouge)

```bash
python -m crypto_mm.main sim-live \
  --symbol BTC/USD \
  --depth 25 \
  --min-size 0.005 \
  --duration 300 \
  --quote-size 0.75 \
  --spread 3 \
  --k-skew 0.80 \
  --kpi-window-sec 120 \
  --latency-ms 8 \
  --priority 0.94 \
  --logs-dir logs
```

> Sorties CSV créées dans `logs/`: `quotes.csv`, `trades.csv`, `pnl.csv`, `kpi_spread.csv`.

---

# 5) Mode Replay (offline, déterministe)

**Rejouer un dump JSONL d’exemple (vitesse x10)**

```bash
python -m crypto_mm.sim.replay \
  --file examples/replay/sample.jsonl \
  --speed 10 \
  --tick 0.1 \
  --quote-size 0.25 \
  --spread-usd 1.0 \
  --k-skew 0.05 \
  --min-quote-size 0.0 \
  --seed-mid 100.0
```

**Rejouer un dump perso**

```bash
python -m crypto_mm.sim.replay \
  --file path/to/your_dump.jsonl \
  --speed 1 \
  --tick 0.1 \
  --quote-size 0.5 \
  --spread-usd 5 \
  --k-skew 0.2 \
  --min-quote-size 0.01
```

> Format attendu (ordre par `ts`) :
>
> ```
> {"type":"book","ts":10.0,"best_bid":99.9,"best_ask":100.1}
> {"type":"trade","ts":11.0,"price":100.60,"qty":0.25}
> ...
> ```
---
