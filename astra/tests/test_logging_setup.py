"""
Regression test for a real bug caught from a live deployment log:
logging.LoggerAdapter's default process() REPLACES the `extra` dict passed
at a log call site with the adapter's own constructor-time `self.extra`,
rather than merging them. Since get_logger() is used throughout Astra both
with constructor-time context (get_logger("app.symbol_worker", symbol=symbol))
and per-call detail (logger.info(msg, extra={"extra_fields": {...}})), this
silently dropped whichever one wasn't captured at construction time -- for
a plain get_logger(name) with no constructor kwargs, EVERY per-call detail
was lost; for one with constructor kwargs, every per-call detail was lost
but the constructor fields survived. Confirmed from a deployment log where
a warning logged with explicit `symbol` and `error` fields showed neither
in the output.
"""
import json
import logging

from app.logging_setup import JsonFormatter, configure_logging, get_logger


def test_per_call_extra_fields_survive_with_no_constructor_context(capsys):
    configure_logging("INFO")
    log = get_logger("test.no_context")
    log.warning("something failed", extra={"extra_fields": {"symbol": "R_100", "error": "boom"}})
    out = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(out)
    assert payload["symbol"] == "R_100"
    assert payload["error"] == "boom"
    assert payload["message"] == "something failed"


def test_constructor_context_and_per_call_fields_both_survive(capsys):
    configure_logging("INFO")
    log = get_logger("test.with_context", symbol="1HZ25V")
    log.info("tick summary", extra={"extra_fields": {"event_type": "tick_summary", "trades_executed": 5}})
    out = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(out)
    assert payload["symbol"] == "1HZ25V"          # constructor-time context
    assert payload["event_type"] == "tick_summary"  # per-call detail
    assert payload["trades_executed"] == 5


def test_constructor_context_alone_still_works_with_no_per_call_extra(capsys):
    configure_logging("INFO")
    log = get_logger("test.context_only", symbol="1HZ25V")
    log.info("plain message")
    out = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(out)
    assert payload["symbol"] == "1HZ25V"


def test_per_call_fields_do_not_leak_across_separate_calls(capsys):
    """A per-call extra_fields dict must not linger and contaminate a later
    call on the same logger instance."""
    configure_logging("INFO")
    log = get_logger("test.no_leak")
    log.info("first", extra={"extra_fields": {"only_on_first": True}})
    log.info("second")
    lines = capsys.readouterr().out.strip().splitlines()
    second_payload = json.loads(lines[-1])
    assert "only_on_first" not in second_payload
