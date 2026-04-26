"""
Structured JSON logging.

Activated by setting ``LOG_JSON=true`` in the environment. When off the
default human-readable text format is preserved.

Each record is emitted as a single-line JSON object with timestamp,
level, logger name, message, and any extras attached via
``logger.info(..., extra={...})``. Exception info is included as a
``exc_info`` field so stack traces remain searchable.

A FastAPI middleware ``trace_id_middleware`` is also exposed so that
each HTTP request gets a UUID stored on ``request.state.trace_id`` and
echoed back in the ``X-Trace-Id`` response header. Downstream callers
can read the trace id via ``current_trace_id()`` to attach it to log
records.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default=""
)


def current_trace_id() -> str:
    return _trace_id_var.get()


_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}


class JSONFormatter(logging.Formatter):
    """Format log records as compact single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_id = current_trace_id()
        if trace_id:
            payload["trace_id"] = trace_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Surface user-provided extras (anything not on the standard LogRecord).
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                value = repr(value)
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Replace handlers on the root logger with one matching the requested format."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
        )
    root.addHandler(handler)


async def trace_id_middleware(request, call_next: Callable[[Any], Awaitable[Any]]):
    """FastAPI middleware that generates / propagates a trace id per request."""
    trace_id = request.headers.get("x-trace-id") or uuid.uuid4().hex
    token = _trace_id_var.set(trace_id)
    try:
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response
    finally:
        _trace_id_var.reset(token)


__all__ = [
    "JSONFormatter",
    "configure_logging",
    "current_trace_id",
    "trace_id_middleware",
]
