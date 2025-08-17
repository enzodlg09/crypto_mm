from __future__ import annotations

from pathlib import Path

import pytest

from crypto_mm.core.config import load_settings


def write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_yaml_then_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "config.yaml"
    write(cfg, "app_name: test-app\nheartbeat_ms: 1000\nlog_level: INFO\nexchange_name: KRAKEN\n")

    env = tmp_path / "example.env"
    write(env, "HEARTBEAT_MS=200\nLOG_LEVEL=DEBUG\n")

    monkeypatch.setenv("CRYPTO_MM_CONFIG", str(cfg))
    monkeypatch.chdir(tmp_path)

    s = load_settings()
    assert s.app_name == "test-app"
    assert s.heartbeat_ms == 200
    assert s.log_level == "DEBUG"


def test_absence_of_env_file_is_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "config.yaml"
    write(cfg, "app_name: test-app\nheartbeat_ms: 600\nexchange_name: KRAKEN\n")
    monkeypatch.setenv("CRYPTO_MM_CONFIG", str(cfg))
    monkeypatch.chdir(tmp_path)

    s = load_settings()
    assert s.heartbeat_ms == 600


def test_invalid_heartbeat_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "config.yaml"
    write(cfg, "heartbeat_ms: 0\n")
    monkeypatch.setenv("CRYPTO_MM_CONFIG", str(cfg))
    monkeypatch.chdir(tmp_path)

    with pytest.raises(Exception):
        _ = load_settings()
