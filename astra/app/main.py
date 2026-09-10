"""
Astra entrypoint.

Startup sequence follows spec section 51:
  1. connect to Deriv (auto-resolving a demo or real account -- see
     app/config.py DerivConfig.use_real_account)
  2. discover symbols: ASTRA_SYMBOLS (currently R_100, 1HZ100V) or, if
     unset, every R_*/1HZ* synthetic index Deriv offers
  3. seed rolling state from recent tick history where available
  4. subscribe to live ticks for every symbol
  5. spawn one independent worker task per symbol (no cross-symbol blocking)
  6. each worker: observe -> update state -> predict (all 3 architectures)
     -> evaluate -> (maybe) trade with the champion architecture -> repeat

Monitoring dashboard/alerting (spec section 30) is intentionally out of
scope for this build -- see app/logging_setup.py.
"""
from __future__ import annotations

import asyncio
import signal

from app.config import get_config
from app.logging_setup import configure_logging, get_logger
from app.tick_summary import TickSummaryTracker
from database.repository import Repository
from database.supabase_client import make_supabase_client
from decision.decision_engine import ARCHITECTURES, DecisionEngine
from execution.orders import execute_decision
from ingestion.deriv_client import DerivClient
from learning.architecture_competition import ArchitectureCompetitionManager
from learning.retraining import RetrainingController
from risk.risk_engine import RiskEngine
from risk.staking import StakingEngine
from state.rolling_state import StateManager

logger = get_logger("app.main")

PREDICTION_LOG_SAMPLE_EVERY_N = 20   # log a NO_TRADE prediction row this often, to keep DB volume sane
STATE_SNAPSHOT_EVERY_N_TICKS = 200
MODEL_PERF_LOG_EVERY_N_TICKS = 500
TICK_SUMMARY_LOG_EVERY = 200         # log a rolling trade/no-trade-reason summary this often, per symbol
TICK_SUMMARY_WINDOW = 500            # ...covering (roughly) this many of the most recent ticks
BALANCE_REFRESH_EVERY_N_TICKS = 200


async def symbol_worker(symbol: str, client: DerivClient, state_manager: StateManager,
                         competition: ArchitectureCompetitionManager, decision_engine: DecisionEngine,
                         repo: Repository, risk_engine: RiskEngine, staking: StakingEngine,
                         retraining: RetrainingController, cfg) -> None:
    queue = await client.subscribe_ticks(symbol)
    state = state_manager.get(symbol)
    tick_count = 0
    currency = cfg.currency
    duration = cfg.get("contracts", "duration", default=1)
    duration_unit = cfg.get("contracts", "duration_unit", default="t")

    log = get_logger("app.symbol_worker", symbol=symbol)
    log.info("Worker started")
    tick_summary = TickSummaryTracker(window_size=TICK_SUMMARY_WINDOW, log_every=TICK_SUMMARY_LOG_EVERY)

    while True:
        tick = await queue.get()
        tick_count += 1

        # Learn from the PREVIOUS tick's prediction now that this tick's
        # digit has settled -- across all 3 architectures at once (see
        # ArchitectureCompetitionManager.observe_pending). This must happen
        # BEFORE state.push() below, using state as it was at prediction
        # time, and BEFORE decision_engine.evaluate() overwrites the
        # competition's pending snapshot with a fresh one for the next tick.
        if competition.has_pending():
            competition.observe_pending(state, tick.digit)

        state.push(tick.digit)
        if cfg.get("database", "persist_ticks", default=True):
            repo.insert_tick(symbol, tick.epoch, tick.quote, tick.digit)

        if tick_count % BALANCE_REFRESH_EVERY_N_TICKS == 0:
            try:
                balance = await client.get_balance()
                if balance and "balance" in balance:
                    risk_engine.set_equity(float(balance["balance"]))
            except Exception as exc:  # noqa: BLE001
                log.warning("Balance refresh failed", extra={"extra_fields": {"error": str(exc)}})

        stake = staking.current_stake(symbol)
        risk_ok, risk_reason = risk_engine.check(stake)

        # This call also runs predict_all() + stashes a fresh pending
        # snapshot (all 3 architectures) for the NEXT tick's observe_pending
        # -- see decision_engine.py::DecisionEngine.evaluate() for why this
        # happens unconditionally, before any trade-eligibility gating.
        decision = await decision_engine.evaluate(
            client, state, competition, stake=stake, currency=currency, risk_ok=risk_ok, risk_reason=risk_reason,
        )

        should_log_prediction = decision.decision != "NO_TRADE" or tick_count % PREDICTION_LOG_SAMPLE_EVERY_N == 0
        prediction_id = repo.insert_prediction(decision) if should_log_prediction else None

        if decision.decision != "NO_TRADE" and risk_ok:
            log.info("Executing trade", extra={"extra_fields": {
                "decision": decision.decision, "reason": decision.reason, "quality": decision.quality_score,
                "architecture": decision.architecture,
            }})
            # Claim a concurrent-trade slot synchronously (no `await` between
            # the risk_ok check above and this reservation), then always
            # release it once settled/failed -- see RiskEngine.reserve_trade_slot
            # docstring for why the ordering matters under asyncio.
            risk_engine.reserve_trade_slot()
            try:
                trade_result = await execute_decision(
                    client, decision, currency=currency, duration=duration, duration_unit=duration_unit,
                    dry_run=cfg.dry_run,
                )
            finally:
                risk_engine.release_trade_slot()

            if trade_result is not None:
                repo.insert_trade(trade_result, prediction_id)
                if trade_result.pnl is not None:
                    risk_engine.record_trade_result(trade_result.pnl)
                    staking.record_result(symbol, bool(trade_result.won))
                if trade_result.error:
                    log.warning("Trade did not settle cleanly", extra={"extra_fields": {"error": trade_result.error}})
                tick_summary.record_trade(trade_result.won, trade_result.pnl)
            else:
                # decided to trade, but no order could even be placed (e.g.
                # quote unavailable at execution time) -- still an attempt,
                # not a decision-engine "no trade", so it's still counted as
                # a trade with an unknown outcome for the summary below.
                tick_summary.record_trade(None, None)
        elif decision.decision != "NO_TRADE" and not risk_ok:
            log.info("Trade blocked by risk engine", extra={"extra_fields": {"reason": risk_reason}})
            repo.insert_risk_event(symbol, "trade_blocked", {"reason": risk_reason, "decision": decision.decision})
            tick_summary.record_risk_blocked(risk_reason)
        else:
            tick_summary.record_no_trade(decision.reason)

        if tick_summary.due():
            summary = tick_summary.build(competition.champion, state.total_observed)
            log.info("Tick summary", extra={"extra_fields": {"event_type": "tick_summary", **summary.__dict__}})
            repo.insert_system_event("app.symbol_worker", "tick_summary", {"symbol": symbol, **summary.__dict__})

        retraining.maybe_retrain(symbol, competition.global_pipeline.registry)

        if tick_count % STATE_SNAPSHOT_EVERY_N_TICKS == 0:
            gp = competition.global_pipeline
            repo.save_symbol_state(
                symbol, state.total_observed, list(state.digits)[-2000:],
                {k: v.tolist() for k, v in gp.champion_weights.items()},
                {k: v.tolist() for k, v in gp.performance.current_weights().items()},
            )

        if tick_count % MODEL_PERF_LOG_EVERY_N_TICKS == 0:
            gp = competition.global_pipeline
            for name in gp.registry.models:
                repo.insert_model_performance(
                    symbol, name, gp.performance.rolling_log_loss(name), gp.champion_weights.get(name),
                )
            for arch in ARCHITECTURES:
                summary = competition.metrics[arch].summary()
                cal = competition.calibration[arch]
                summary["calibration"] = (cal["over"].quality_score() + cal["under"].quality_score()) / 2.0
                repo.insert_architecture_performance(symbol, arch, summary)


async def discover_symbols(client: DerivClient, cfg) -> list[str]:
    if cfg.symbol_override:
        logger.info("Using ASTRA_SYMBOLS override", extra={"extra_fields": {"symbols": cfg.symbol_override}})
        return cfg.symbol_override
    prefixes = cfg.get("symbols", "prefixes", default=["R_", "1HZ"])
    symbols = await client.get_active_synthetic_symbols(prefixes)
    logger.info("Discovered symbols", extra={"extra_fields": {"count": len(symbols), "symbols": symbols}})
    return symbols


async def seed_symbol(client: DerivClient, state_manager: StateManager, symbol: str,
                       repo: Repository, count: int = 2000) -> None:
    try:
        history = await client.get_history(symbol, count=count)
        state_manager.seed(symbol, [t.digit for t in history])
        logger.info("Seeded symbol history", extra={"extra_fields": {"symbol": symbol, "n": len(history)}})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Seeding failed, will build state from live ticks only",
                        extra={"extra_fields": {"symbol": symbol, "error": str(exc)}})
        # Also persisted to Supabase, not just the Railway log stream -- so
        # this is diagnosable even from a log export that misses the
        # startup window (seeding happens once, in the first couple of
        # seconds, and is easy to miss when grabbing "the last N minutes").
        repo.insert_system_event("app.seed_symbol", "seeding_failed", {"symbol": symbol, "error": str(exc)})


async def main() -> None:
    cfg = get_config()
    configure_logging(cfg.log_level)

    account_kind = "REAL MONEY" if cfg.deriv.use_real_account else "demo"
    logger.info("Starting Astra", extra={"extra_fields": {"dry_run": cfg.dry_run, "account_kind": account_kind}})
    if cfg.deriv.use_real_account and cfg.dry_run:
        logger.warning("use_real_account=True with dry_run=True: connecting to the REAL account "
                        "but no orders will be placed.")
    elif cfg.deriv.use_real_account:
        logger.warning("use_real_account=True: this run trades REAL MONEY.")
    else:
        logger.info(f"Trading against the DEMO account (dry_run={cfg.dry_run}).")

    supabase = make_supabase_client(cfg.supabase.url, cfg.supabase.service_key, cfg.supabase.enabled)
    repo = Repository(supabase, persist_ticks=cfg.get("database", "persist_ticks", default=True))

    client = DerivClient(
        app_id=cfg.deriv.app_id, api_token=cfg.deriv.api_token,
        ws_url=cfg.deriv.ws_url, options_token_url=cfg.deriv.options_token_url,
        account_id=cfg.deriv.account_id, use_real_account=cfg.deriv.use_real_account,
    )
    await client.connect()
    repo.insert_system_event("app.main", "startup")

    symbols = await discover_symbols(client, cfg)
    if not symbols:
        logger.error("No symbols discovered -- nothing to trade. Check DERIV_APP_ID/token permissions.")
        return

    max_window = max(cfg.get("feature_windows", default=[2500]))
    state_manager = StateManager(max_window=max_window, max_markov_order=cfg.get("max_markov_order", default=3))

    risk_cfg = cfg.get("risk", default={})
    risk_engine = RiskEngine(
        base_stake=risk_cfg.get("base_stake", 1.0), max_stake=risk_cfg.get("max_stake", 5.0),
        max_consecutive_losses=risk_cfg.get("max_consecutive_losses", 5),
        max_daily_loss=risk_cfg.get("max_daily_loss", 25.0), max_drawdown=risk_cfg.get("max_drawdown", 40.0),
        max_trades_per_day=risk_cfg.get("max_trades_per_day", 500),
        cooldown_seconds_after_max_losses=risk_cfg.get("cooldown_seconds_after_max_losses", 900),
        max_concurrent_trades=risk_cfg.get("max_concurrent_trades", 2),
    )
    staking_cfg = risk_cfg.get("staking", {})
    staking = StakingEngine(
        base_stake=risk_cfg.get("base_stake", 1.0), enabled=staking_cfg.get("enabled", False),
        progression_factor=staking_cfg.get("progression_factor", 2.0),
        max_steps=staking_cfg.get("max_steps", 3), max_stake=risk_cfg.get("max_stake", 5.0),
    )

    retrain_cfg = cfg.get("retraining", default={})
    retraining = RetrainingController(
        every_n=retrain_cfg.get("batch_model_every_n_observations", 300),
        min_observations=retrain_cfg.get("min_observations_for_batch_models", 500),
    )

    decision_engine = DecisionEngine(cfg)

    seed_tasks = [seed_symbol(client, state_manager, s, repo) for s in symbols]
    await asyncio.gather(*seed_tasks)

    workers = []
    for symbol in symbols:
        repo.upsert_symbol(symbol)
        competition = ArchitectureCompetitionManager(symbol, cfg, repository=repo)
        workers.append(asyncio.create_task(
            symbol_worker(symbol, client, state_manager, competition, decision_engine, repo,
                          risk_engine, staking, retraining, cfg),
            name=f"worker-{symbol}",
        ))

    stop_event = asyncio.Event()

    def _handle_signal():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            pass  # not available on some platforms (e.g. Windows)

    await stop_event.wait()

    for w in workers:
        w.cancel()
    await client.close()
    repo.insert_system_event("app.main", "shutdown")


if __name__ == "__main__":
    asyncio.run(main())
