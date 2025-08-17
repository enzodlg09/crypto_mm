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

    def __call__(self) -> Dict[str, Any]:
        path = _default_config_path()
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}


class Settings(BaseSettings):
    """
    Load settings in order (later overrides earlier):
    1) YAML (config.yaml / CRYPTO_MM_CONFIG)
    2) example.env
    3) Environment variables
    4) Init kwargs
    """

    model_config = SettingsConfigDict(env_file="example.env", env_file_encoding="utf-8", extra="forbid")

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
        return (YamlSettingsSource(settings_cls), dotenv_settings, env_settings, init_settings, file_secret_settings)


def load_settings() -> Settings:
    """Load & validate settings."""
    return Settings()
