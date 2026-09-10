"""
Tick-by-tick replay simulator.

SCOPE NOTE: this replays only the "global" architecture (SymbolPipeline),
not the full 3-way architecture competition added in
learning/architecture_competition.py -- it's still useful for sanity-
checking the core feature/model/decision pipeline offline (does it abstain
on noise, does it trade and win on an injected bias), just not for
comparing the three architectures against each other. That comparison
already happens live, continuously, in production via shadow evaluation;
extending this offline replay to cover it too is a reasonable follow-up but
wasn't in scope for this pass.

Reuses the exact same SymbolPipeline / DecisionEngine code paths as live
trading -- the only difference is that contract quotes are synthesized from
a configurable assumed payout instead of fetched from Deriv (there's no live
proposal to query in a replay), and no real orders are placed. This
guarantees no look-ahead: predictions at tick t are built only from
`state.window(...)` as of tick t, and the outcome used for `observe()` is
always the very next tick in the replayed sequence.

Usage:
    python -m backtest.simulator --digits path/to/digits.json --symbol R_100
or import `replay()` directly and pass a list[int] of digits (e.g. pulled
from astra_ticks via database/repository.py for a symbol/time range).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from app.config import get_config
from decision.decision_engine import SymbolPipeline
from models.ensemble import EVEN_DIGITS, ODD_DIGITS, combine
from pricing.edge import compute_edge
from pricing.payout import ContractQuote
from pricing.mispricing import check_mispricing
from state.rolling_state import StateManager


@dataclass
class SimResult:
    n_ticks: int
    n_trades: int
    wins: int
    losses: int
    total_pnl: float
    final_champion_log_loss: float | None
    final_challenger_log_loss: float | None


def _synthetic_quote(symbol: str, contract_type: str, stake: float, assumed_payout_ratio: float) -> ContractQuote:
    """Approximates a Deriv payout for offline replay, WITHOUT a live proposal.

    Unlike the old DIGITOVER/DIGITUNDER version of this function, a flat
    assumed payout is actually a reasonable approximation for DIGITEVEN/
    DIGITODD: both sides are an exact 50/50 split of the 10 digits (5 each),
    so there's no barrier-dependent base-rate geometry to misrepresent --
    the concern that made the old Over/Under version's flat-payout backtests
    potentially misleading (a barrier of 2 has a ~70% win probability baked
    in regardless of any real edge) simply doesn't apply here. Still: real
    Deriv payouts fluctuate with market conditions even for a fixed 50/50
    contract, so treat backtest P&L as a sanity check that the wiring
    (features -> models -> decision -> settlement) behaves as intended, not
    as a precise estimate of real returns. Live trading never uses this
    function -- pricing/payout.py always fetches a real proposal from Deriv
    before every decision and again before every buy.
    """
    payout = stake * (1 + assumed_payout_ratio)
    return ContractQuote(symbol=symbol, contract_type=contract_type, stake=stake,
                          payout=payout, ask_price=stake, proposal_id=None, spot=None)


def replay(symbol: str, digits: list[int], cfg=None, assumed_payout_ratio: float = 0.9,
           stake: float = 1.0) -> SimResult:
    cfg = cfg or get_config()
    min_samples = cfg.get("min_samples_per_symbol", default=300)
    mp_cfg = cfg.get("mispricing", default={})

    max_window = max(cfg.get("feature_windows", default=[2500]))
    max_markov_order = cfg.get("max_markov_order", default=3)
    state_manager = StateManager(max_window=max_window, max_markov_order=max_markov_order)
    state = state_manager.get(symbol)
    pipeline = SymbolPipeline(symbol, cfg)

    n_trades = wins = losses = 0
    total_pnl = 0.0
    pending = None

    for i, digit in enumerate(digits):
        if pending is not None:
            predictions, bundle, _even_p, _odd_p = pending
            pipeline.observe(state, bundle, predictions, digit)

        state.push(digit)

        if not state.has_min_samples(min_samples):
            predictions, bundle = pipeline.predict(state)
            pending = (predictions, bundle, 0.0, 0.0)
            continue

        predictions, bundle = pipeline.predict(state)
        ensemble_vec = combine(predictions, pipeline.champion_weights)
        raw_even = float(np.sum(ensemble_vec[list(EVEN_DIGITS)]))
        raw_odd = float(np.sum(ensemble_vec[list(ODD_DIGITS)]))
        cal_even = pipeline.calibration_even.calibrate(raw_even)
        cal_odd = pipeline.calibration_odd.calibrate(raw_odd)
        pipeline._pending_even_prob = raw_even
        pipeline._pending_odd_prob = raw_odd

        quote_even = _synthetic_quote(symbol, "DIGITEVEN", stake, assumed_payout_ratio)
        quote_odd = _synthetic_quote(symbol, "DIGITODD", stake, assumed_payout_ratio)
        edge_even = compute_edge(quote_even, cal_even)
        edge_odd = compute_edge(quote_odd, cal_odd)

        best = None
        for side, edge_result, cal_score in (
            ("EVEN", edge_even, pipeline.calibration_even.quality_score()),
            ("ODD", edge_odd, pipeline.calibration_odd.quality_score()),
        ):
            check = check_mispricing(
                edge_result, sample_size=state.total_observed, calibration_score=cal_score,
                model_agreement=0.8,  # not recomputed in this lightweight replay
                minimum_edge=mp_cfg.get("minimum_edge", 0.03),
                minimum_probability=mp_cfg.get("minimum_probability", 0.55),
                minimum_calibration_score=mp_cfg.get("minimum_calibration_score", 0.6),
                minimum_model_agreement=0.0,
                minimum_sample_size=mp_cfg.get("minimum_sample_size", 300),
            )
            if check.passes and (best is None or edge_result.expected_value > best[1].expected_value):
                best = (side, edge_result)

        if i + 1 < len(digits) and best is not None:
            side, edge_result = best
            next_digit = digits[i + 1]
            won = (next_digit % 2 == 0) if side == "EVEN" else (next_digit % 2 == 1)
            n_trades += 1
            pnl = edge_result.quote.payout - stake if won else -stake
            total_pnl += pnl
            wins += int(won)
            losses += int(not won)

        pending = (predictions, bundle, raw_even, raw_odd)

    return SimResult(
        n_ticks=len(digits), n_trades=n_trades, wins=wins, losses=losses, total_pnl=total_pnl,
        final_champion_log_loss=(
            float(np.mean(pipeline._ensemble_logloss_champion)) if pipeline._ensemble_logloss_champion else None
        ),
        final_challenger_log_loss=(
            float(np.mean(pipeline._ensemble_logloss_challenger)) if pipeline._ensemble_logloss_challenger else None
        ),
    )


def _main():
    parser = argparse.ArgumentParser(description="Replay a digit sequence through Astra's pipeline")
    parser.add_argument("--digits", required=True, help="Path to a JSON file containing a list of ints")
    parser.add_argument("--symbol", default="R_100")
    parser.add_argument("--stake", type=float, default=1.0)
    args = parser.parse_args()

    print("NOTE: this replay uses a flat placeholder payout for a 50/50 "
          "DIGITEVEN/DIGITODD split -- see the _synthetic_quote docstring. "
          "Treat P&L here as a pipeline sanity check, not a precise "
          "performance estimate.\n")

    with open(args.digits) as f:
        digits = json.load(f)

    result = replay(args.symbol, digits, stake=args.stake)
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    _main()
