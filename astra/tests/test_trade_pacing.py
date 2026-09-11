import time

from risk.risk_engine import RiskEngine


def make_engine(**overrides):
    kwargs = dict(
        base_stake=1.0, max_stake=5.0, max_consecutive_losses=5,
        max_daily_loss=25.0, max_drawdown=40.0, max_trades_per_day=500,
        cooldown_seconds_after_max_losses=900, max_concurrent_trades=2,
        min_seconds_between_trades=5.0,
    )
    kwargs.update(overrides)
    return RiskEngine(**kwargs)


def test_second_trade_is_blocked_before_the_pacing_window_elapses():
    """Regression test: a live deployment's trade table showed new positions
    opening 1-2 seconds after the previous one settled, repeatedly -- with no
    general pacing gate, only a punitive post-loss cooldown, the bot could
    fire on every single eligible tick back-to-back."""
    engine = make_engine(min_seconds_between_trades=5.0)
    ok, reason = engine.check(stake=1.0)
    assert ok
    engine.reserve_trade_slot()
    engine.release_trade_slot()  # contract settled

    ok, reason = engine.check(stake=1.0)
    assert not ok
    assert reason == "trade_pacing_cooldown"


def test_trade_allowed_again_once_pacing_window_elapses():
    engine = make_engine(min_seconds_between_trades=0.05)
    ok, _ = engine.check(stake=1.0)
    assert ok
    engine.reserve_trade_slot()
    engine.release_trade_slot()

    ok, reason = engine.check(stake=1.0)
    assert not ok
    assert reason == "trade_pacing_cooldown"

    time.sleep(0.06)
    ok, reason = engine.check(stake=1.0)
    assert ok
    assert reason is None


def test_pacing_disabled_when_min_seconds_is_zero():
    engine = make_engine(min_seconds_between_trades=0.0)
    engine.reserve_trade_slot()
    engine.release_trade_slot()
    ok, reason = engine.check(stake=1.0)
    assert ok
    assert reason is None


def test_pacing_is_independent_of_the_punitive_loss_cooldown():
    """min_seconds_between_trades and cooldown_seconds_after_max_losses are
    two different mechanisms -- a fast pacing window shouldn't trip the
    punitive cooldown, and vice versa."""
    engine = make_engine(min_seconds_between_trades=0.0, max_consecutive_losses=2,
                          cooldown_seconds_after_max_losses=900)
    engine.reserve_trade_slot()
    engine.release_trade_slot()
    engine.record_trade_result(pnl=-1.0)
    engine.reserve_trade_slot()
    engine.release_trade_slot()
    engine.record_trade_result(pnl=-1.0)  # 2nd consecutive loss -> punitive cooldown trips

    ok, reason = engine.check(stake=1.0)
    assert not ok
    assert reason == "max_consecutive_losses_reached"


def test_pacing_check_does_not_require_a_previous_trade():
    """The very first trade of a run must not be blocked -- there's no
    last_trade_time yet."""
    engine = make_engine(min_seconds_between_trades=5.0)
    ok, reason = engine.check(stake=1.0)
    assert ok
    assert reason is None
