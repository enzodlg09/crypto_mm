# crypto-mm (Kraken)

Scaffold MM Python (typé, modulaire) avec WebSocket **Kraken Spot v1** (trades + order book L2 incrémental + checksum CRC32).

- Config hiérarchique: `config.yaml` (défaut) + `.env` (overrides) via **pydantic-settings**
- CLI (rafraîchit/TTY 500 ms): `live`, `replay`, `plot`, `check`, `ws-ping`, `ws-trades`, `ws-book`
- Logging structuré **JSON** (levels INFO/ERROR, latences p50/p95)
- Tests **pytest** (unitaires + 1 intégration)
- Conventions: timestamps UTC ISO8601; montants **USD**; quantités **BTC**; prix **USD/BTC**

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Aide
python -m crypto_mm.main --help

# Sanity checks (offline + petits tests live)
python -m crypto_mm.main check
python -m crypto_mm.main ws-ping                         # ping v1 Kraken
python -m crypto_mm.main ws-trades --pair XBT/USD --duration 5
python -m crypto_mm.main ws-book   --pair XBT/USD --duration 5 --depth 10

# Démos horloge/replay/plot
python -m crypto_mm.main live --duration 5
python -m crypto_mm.main replay --duration 3
python -m crypto_mm.main plot
