"""
Structured logging for Astra.

Per the current build scope, the monitoring dashboard/alerting subsystem
(spec section 30) is intentionally OUT of scope -- Railway's log viewer plus
the Supabase tables (astra_predictions, astra_trades, astra_regime_log,
astra_system_events) are the source of truth for now. This module just makes
sure every log line is structured JSON so it's easy to grep/parse later, and
gives every component its own named logger.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


class ExtraFieldsAdapter(logging.LoggerAdapter):
    """logging.LoggerAdapter's default process() REPLACES whatever `extra`
    dict is passed at the call site with the adapter's own constructor-time
    `self.extra`, rather than merging them. That silently dropped every
    per-call `extra={"extra_fields": {...}}` passed to a logger obtained via
    get_logger() -- confirmed from a live deployment log where a warning
    ("Seeding failed...") was logged with an explicit `error` and `symbol`
    detail that never made it to the output, showing only the adapter's own
    base fields. This subclass merges the two instead, so both the
    logger's own context (e.g. `symbol=...` set once at get_logger() time)
    and whatever a specific call passes are preserved."""

    def process(self, msg, kwargs):
        call_extra = kwargs.get("extra") or {}
        merged_fields = {
            **self.extra.get("extra_fields", {}),
            **call_extra.get("extra_fields", {}),
        }
        kwargs["extra"] = {"extra_fields": merged_fields} if merged_fields else {}
        return msg, kwargs


def get_logger(name: str, **extra_fields) -> logging.LoggerAdapter:
    logger = logging.getLogger(name)
    return ExtraFieldsAdapter(logger, {"extra_fields": extra_fields} if extra_fields else {})
