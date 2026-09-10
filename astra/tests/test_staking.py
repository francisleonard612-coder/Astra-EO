from risk.staking import StakingEngine, _round_stake


def _has_at_most_2_decimals(x: float) -> bool:
    return round(x, 2) == x


def test_flat_stake_is_returned_unchanged_when_progression_disabled():
    engine = StakingEngine(base_stake=1.0, enabled=False, progression_factor=2.0, max_steps=3, max_stake=5.0)
    assert engine.current_stake("SYM") == 1.0


def test_progression_never_produces_more_than_2_decimal_places():
    """Regression test: a live deployment log showed ~1/3 of Deriv proposal
    requests failing with 'Stake can not have more than 2 decimal places.'
    once martingale was enabled with a non-power-of-two factor (1.15):
    1.0 * 1.15**2 == 1.3225. Every step, across a range of realistic
    factors, must now round cleanly."""
    for factor in (1.15, 1.07, 1.33, 2.0, 1.5):
        engine = StakingEngine(base_stake=1.0, enabled=True, progression_factor=factor, max_steps=5, max_stake=100.0)
        state = engine._get("SYM")
        for step in range(6):
            state.step = step
            stake = engine.current_stake("SYM")
            assert _has_at_most_2_decimals(stake), f"factor={factor} step={step} produced {stake}"


def test_progression_rounds_down_never_up():
    """Rounding must never push a stake ABOVE what the progression formula
    computed -- that would be a risk-limit bypass, however small."""
    engine = StakingEngine(base_stake=1.0, enabled=True, progression_factor=1.15, max_steps=5, max_stake=100.0)
    state = engine._get("SYM")
    state.step = 2  # raw value: 1.0 * 1.15**2 == 1.3225
    stake = engine.current_stake("SYM")
    assert stake == 1.32
    assert stake <= 1.3225


def test_progression_still_respects_max_stake_after_rounding():
    engine = StakingEngine(base_stake=1.0, enabled=True, progression_factor=3.0, max_steps=10, max_stake=5.0)
    state = engine._get("SYM")
    state.step = 5  # raw value would be far above max_stake
    stake = engine.current_stake("SYM")
    assert stake <= 5.0
    assert _has_at_most_2_decimals(stake)


def test_round_stake_helper_floors_rather_than_rounds_nearest():
    assert _round_stake(1.3225) == 1.32
    assert _round_stake(1.999) == 1.99
    assert _round_stake(1.0) == 1.0
