"""JSONFormatter emits one valid JSON object per line, merging `extra` fields."""

import json
import logging

from inference_server.logging_config import JSONFormatter


def _format(record):
    return JSONFormatter().format(record)


def test_json_formatter_basic_fields():
    rec = logging.LogRecord("test.logger", logging.INFO, __file__, 1, "hello", (), None)
    out = json.loads(_format(rec))
    assert out["level"] == "INFO"
    assert out["logger"] == "test.logger"
    assert out["msg"] == "hello"
    assert "ts" in out


def test_json_formatter_merges_extra():
    rec = logging.LogRecord("l", logging.INFO, __file__, 1, "http_request", (), None)
    rec.endpoint = "/generate"          # what logger.info(..., extra={...}) attaches
    rec.duration_ms = 12.3
    out = json.loads(_format(rec))
    assert out["endpoint"] == "/generate"
    assert out["duration_ms"] == 12.3


def test_json_formatter_includes_exception():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        rec = logging.LogRecord("l", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    out = json.loads(_format(rec))
    assert "boom" in out["exc"]
