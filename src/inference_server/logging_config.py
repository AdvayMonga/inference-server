"""Logging setup — human-readable text (dev) or structured JSON lines (prod/Modal).

JSON logs are machine-parseable for log aggregators. `logger.info("msg", extra={...})` fields are
merged into the JSON object, so call sites can attach structured context (session_id, latency, etc.)
without string-formatting. Toggle via `LOG_FORMAT=json|text` (default text).
"""

from __future__ import annotations

import json
import logging

# LogRecord attributes that are framework-internal — everything else in __dict__ is a user `extra`.
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"taskName", "message", "asctime"}


class JSONFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, msg, any `extra` fields, exception."""

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                out[key] = val
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup_logging(fmt: str = "text", level: int = logging.INFO) -> None:
    """Install a single stream handler on the root logger. `force`-style: replaces existing handlers."""
    handler = logging.StreamHandler()
    if fmt == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
