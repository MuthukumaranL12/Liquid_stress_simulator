import os
from simulator import Simulator
from stress_events import StressEventManager


def test_simulate_days_columns():
    sim = Simulator(seed=0)
    df = sim.simulate_days(5)
    assert len(df) == 5
    expected_cols = {"day", "cash_reserve", "total_deposits", "loan_assets", "liquidity_ratio"}
    assert expected_cols.issubset(set(df.columns))


def test_stress_event_manager_keys():
    m = StressEventManager(seed=0)
    mods = m.maybe_trigger(1, {})
    allowed = {"withdrawals", "defaults", "deposits"}
    assert set(mods.keys()).issubset(allowed)
