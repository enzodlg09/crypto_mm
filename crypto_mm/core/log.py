from __future__ import annotations

import json
import logging
from typing import Any, Protocol


class LoggerLike(Protocol):
    def info(self, msg: Any) -> None: ...
    def error(self, msg: Any) -> None: ...
    def debug(self, msg: Any) -> None: ...


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.msg, dict):
            payload = record.msg
        else:
            payload = {"message": str(record.msg)}
        payload.setdefault("level", record.levelname)
        payload.setdefault("logger", record.name)
        return json.dumps(payload, separators=(",", ":"))


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger
