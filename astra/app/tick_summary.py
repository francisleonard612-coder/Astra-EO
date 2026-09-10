"""
Per-symbol ROLLING tick-outcome summary, logged (and persisted to
astra_system_events) every `log_every` ticks so a Railway log skim answers
"is Astra actually trading, and if not, why not" without having to query
Supabase or manually parse decision.reason strings.

This is a trailing window, not a hard-reset window and not a lifetime
cumulative total:

- A hard-reset window (the original 150-tick design) throws away all
  context every time it logs, so each line only ever shows a short,
  noisy slice and you have to add several of them up in your head to see
  a trend.
- A lifetime cumulative total (the previous version of this file) never
  forgets anything, so a reason that was common only in the first few
  minutes keeps inflating every later summary forever, even long after
  it stopped happening.

Instead, `window_size` ticks of history are kept in a deque and every
`build()` call recomputes counts from whatever's currently in it. Ticks
older than `window_size` age out on their own as new ones arrive, so a
summary always reflects "roughly the last `window_size` ticks", which is
long enough to smooth over single-tick noise but short enough that stale
behavior fades out instead of accumulating forever. `log_every` (how often
a summary actually gets logged) is independent of `window_size` (how much
history each summary covers) -- e.g. log every 200 ticks, but each log
line can cover the trailing 500 for a bit more smoothing.
"""
from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass, field

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
    window_ticks: int
    trades_executed: int
    wins: int
    losses: int
    pnl: float
    no_trade_ticks: int
    top_no_trade_reasons: dict[str, int]
    champion_architecture: str
    sample_size: int


@dataclass
class _TickRecord:
    """One tick's outcome, as stored in the rolling window."""
    is_trade: bool
    won: bool | None = None
    pnl: float | None = None
    reasons: tuple[str, ...] = ()


class TickSummaryTracker:
    """One instance per symbol. Feed it every tick's outcome via
    `record_trade` / `record_risk_blocked` / `record_no_trade`; check
    `due()` after each tick and call `build()` when it fires.

    `window_size` bounds how much history is kept (older ticks age out
    automatically); `log_every` controls how often `due()` fires. Nothing
    is ever explicitly reset -- the deque's maxlen does that job.
    """

    def __init__(self, window_size: int = 500, log_every: int = 200):
        self.window_size = window_size
        self.log_every = log_every
        self._records: deque[_TickRecord] = deque(maxlen=window_size)
        self.total_ticks = 0  # lifetime tick count, only used to time due()

    def record_no_trade(self, reason: str) -> None:
        self.total_ticks += 1
        self._records.append(_TickRecord(is_trade=False, reasons=tuple(extract_no_trade_reasons(reason))))

    def record_risk_blocked(self, risk_reason: str | None) -> None:
        self.total_ticks += 1
        self._records.append(_TickRecord(is_trade=False, reasons=(f"risk_blocked:{risk_reason}",)))

    def record_trade(self, won: bool | None, pnl: float | None) -> None:
        self.total_ticks += 1
        self._records.append(_TickRecord(is_trade=True, won=won, pnl=pnl))

    def due(self) -> bool:
        return self.total_ticks > 0 and self.total_ticks % self.log_every == 0

    def build(self, champion_architecture: str, sample_size: int) -> TickSummary:
        trades = wins = losses = 0
        pnl_sum = 0.0
        no_trade_reasons: Counter = Counter()
        for rec in self._records:
            if rec.is_trade:
                trades += 1
                if rec.won is True:
                    wins += 1
                elif rec.won is False:
                    losses += 1
                if rec.pnl is not None:
                    pnl_sum += rec.pnl
            else:
                for token in rec.reasons:
                    no_trade_reasons[token] += 1

        window_ticks = len(self._records)
        return TickSummary(
            window_ticks=window_ticks,
            trades_executed=trades,
            wins=wins,
            losses=losses,
            pnl=round(pnl_sum, 4),
            no_trade_ticks=window_ticks - trades,
            top_no_trade_reasons=dict(no_trade_reasons.most_common(5)),
            champion_architecture=champion_architecture,
            sample_size=sample_size,
        )
