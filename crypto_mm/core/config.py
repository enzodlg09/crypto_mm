from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


def _default_config_path() -> str:
    return os.environ.get("CRYPTO_MM_CONFIG", "config.yaml")


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Load top-level keys from a YAML file (path via CRYPTO_MM_CONFIG or ./config.yaml)."""

    def __init__(self, settings_cls):
        super().__init__(settings_cls)
        self._data: Dict[str, Any] | None = None

    def _load(self) -> Dict[str, Any]:
        path = _default_config_path()
        if not os.path.exists(path):
            self._data = {}
            return self._data
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        self._data = data
        return data

    # required by pydantic-settings v2
    def get_field_value(self, field, field_name):  # type: ignore[override]
        data = self._data if self._data is not None else self._load()
        if field_name in data:
            return data[field_name], field_name, False
        return None, field_name, False

    def __call__(self) -> Dict[str, Any]:
        return self._data if self._data is not None else self._load()


class DotEnvAutoSource(PydanticBaseSettingsSource):
    """
    .env loader with auto-fallback:
    1) CRYPTO_MM_ENV_FILE (if exists)
    2) example.env (if exists)
    3) .env (if exists)

    Keys are mapped by lowercasing (e.g. HEARTBEAT_MS -> heartbeat_ms).
    """

    def __init__(self, settings_cls):
        super().__init__(settings_cls)
        self._data: Dict[str, Any] | None = None

    @staticmethod
    def _select_env_file() -> Optional[str]:
        candidate = os.environ.get("CRYPTO_MM_ENV_FILE")
        if candidate and os.path.exists(candidate):
            return candidate
        if os.path.exists("example.env"):
            return "example.env"
        if os.path.exists(".env"):
            return ".env"
        return None

    def _load(self) -> Dict[str, Any]:
        path = self._select_env_file()
        if not path:
            self._data = {}
            return self._data
        out: Dict[str, Any] = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip().lower()
                    v = v.strip().strip('"').strip("'")
                    out[k] = v
        except Exception:
            out = {}
        self._data = out
        return out

    # required by pydantic-settings v2
    def get_field_value(self, field, field_name):  # type: ignore[override]
        data = self._data if self._data is not None else self._load()
        if field_name in data:
            return data[field_name], field_name, False
        return None, field_name, False

    def __call__(self) -> Dict[str, Any]:
        return self._data if self._data is not None else self._load()


class Settings(BaseSettings):
    """
    Precedence (first wins):
    1) init kwargs
    2) OS environment variables
    3) DotEnvAuto (CRYPTO_MM_ENV_FILE → example.env → .env)
    4) YAML (config.yaml / CRYPTO_MM_CONFIG)
    5) file secrets
    """

    model_config = SettingsConfigDict(
        env_file=".env",              # kept for compatibility (not relied upon)
        env_file_encoding="utf-8",
        extra="forbid",
    )

    app_name: str = Field(default="crypto-mm")
    heartbeat_ms: int = Field(default=500, description="Heartbeat period in milliseconds.")
    log_level: str = Field(default="INFO", description="Logging level (INFO|DEBUG|ERROR).")

    exchange_name: str = Field(default="KRAKEN")
    websocket_url: Optional[str] = Field(default="wss://ws.kraken.com/v2")
    api_key: Optional[str] = Field(default=None)
    api_secret: Optional[str] = Field(default=None)

    data_dir: str = Field(default="data")
    plot_output_dir: str = Field(default="plots")

    @field_validator("heartbeat_ms")
    @classmethod
    def _positive_heartbeat(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("heartbeat_ms must be > 0")
        return v

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        # FIRST WINS (highest priority first)
        return (
            init_settings,                # 1
            env_settings,                 # 2
            DotEnvAutoSource(settings_cls),  # 3
            YamlSettingsSource(settings_cls), # 4
            file_secret_settings,         # 5
        )


def load_settings() -> Settings:
    """Load & validate settings."""
    return Settings()
