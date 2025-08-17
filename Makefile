.PHONY: run test lint

run:
	python -m crypto_mm.main live --duration 5

test:
	python -m pytest -q

lint:
	ruff check crypto_mm tests
