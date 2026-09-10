"""
Per-symbol CUMULATIVE tick-outcome summary, logged (and persisted to
astra_system_events) every `log_every` ticks so a Railway log skim answers
"is Astra actually trading, and if not, why not" without having to query
Supabase or manually parse decision.reason strings.

Counts are never reset: each summary reflects every tick seen since the
worker started, not just the ticks since the previous log line. That way a
single "Tick summary" entry at, say, tick 800 already tells the whole story
(e.g. "0/800 trades, insufficient_edge on 800/800") instead of forcing you to
add up several independent 150-tick windows in your head to see the trend.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_ARCH_PREFIX_RE = re.compile(r"^\[(global|specialist|hybrid)\]\s*")


def extract_no_trade_reasons(reason: str) -> list[str]:
    """A NO_TRADE decision.reason is either a single token
    ("insufficient_sample_size", "no_ready_models", "no_positive_edge") or a
    comma-joined set of gate failures, optionally prefixed with the
    architecture that produced it (e.g. "[global] insufficient_edge,
    probability_below_minimum"). This splits it into individual reason
    tokens so a summary counts how often each SPECIFIC gate is the blocker,
    rather than treating every distinct combination as its own bucket --
    "insufficient_edge appeared in 40/150 ticks" is a debuggable signal;
    "'insufficient_edge,quality_score_below_minimum' appeared in 12/150
    ticks, 'insufficient_edge' alone in 9/150, ..." is not, at a glance.
    """
    reason = _ARCH_PREFIX_RE.sub("", reason or "")
    return [tok.strip() for tok in reason.split(",") if tok.strip()]


@dataclass
class TickSummary:
    total_ticks: int
    trades_executed: int
    wins: int
    losses: int
    pnl: float
    no_trade_ticks: int
    top_no_trade_reasons: dict[str, int]
    champion_architecture: str
    sample_size: int


class TickSummaryTracker:
    """One instance per symbol. Feed it every tick's outcome via
    `record_trade` / `record_risk_blocked` / `record_no_trade`; check
    `due()` after each tick and call `build()` when it fires.

    Unlike a windowed tracker, nothing is ever reset -- `build()` just
    snapshots the running totals. `due()` fires every `log_every` ticks
    (200 by default) purely to control log/DB write frequency; it has no
    effect on what the summary contains.
    """

    def __init__(self, log_every: int = 200):
        self.log_every = log_every
        self.ticks = 0
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.pnl = 0.0
        self.no_trade_reasons: Counter = Counter()

    def record_no_trade(self, reason: str) -> None:
        self.ticks += 1
        for token in extract_no_trade_reasons(reason):
            self.no_trade_reasons[token] += 1

    def record_risk_blocked(self, risk_reason: str | None) -> None:
        self.ticks += 1
        self.no_trade_reasons[f"risk_blocked:{risk_reason}"] += 1

    def record_trade(self, won: bool | None, pnl: float | None) -> None:
        self.ticks += 1
        self.trades += 1
        if won is True:
            self.wins += 1
        elif won is False:
            self.losses += 1
        if pnl is not None:
            self.pnl += pnl

    def due(self) -> bool:
        return self.ticks > 0 and self.ticks % self.log_every == 0

    def build(self, champion_architecture: str, sample_size: int) -> TickSummary:
        return TickSummary(
            total_ticks=self.ticks,
            trades_executed=self.trades,
            wins=self.wins,
            losses=self.losses,
            pnl=round(self.pnl, 4),
            no_trade_ticks=self.ticks - self.trades,
            top_no_trade_reasons=dict(self.no_trade_reasons.most_common(5)),
            champion_architecture=champion_architecture,
            sample_size=sample_size,
        )
