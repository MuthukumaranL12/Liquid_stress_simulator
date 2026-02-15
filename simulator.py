import numpy as np
import pandas as pd
from stress_events import StressEventManager
from typing import Optional


class Simulator:
    def __init__(self,
                 initial_cash: float = 1_000_000.0,
                 initial_deposits: float = 10_000_000.0,
                 loan_assets: float = 5_000_000.0,
                 mean_daily_deposits: float = 100_000.0,
                 std_daily_deposits: float = 20_000.0,
                 mean_daily_withdrawals: float = 95_000.0,
                 std_daily_withdrawals: float = 25_000.0,
                 mean_new_loans: float = 50_000.0,
                 std_new_loans: float = 10_000.0,
                 mean_repayments: float = 40_000.0,
                 std_repayments: float = 8_000.0,
                 mean_defaults: float = 2_000.0,
                 std_defaults: float = 5_000.0,
                 seed: Optional[int] = None,
                 stress_manager: Optional[StressEventManager] = None):
        self.rng = np.random.default_rng(seed)
        self.cash_reserve = float(initial_cash)
        self.total_deposits = float(initial_deposits)
        self.loan_assets = float(loan_assets)

        self.mean_daily_deposits = mean_daily_deposits
        self.std_daily_deposits = std_daily_deposits
        self.mean_daily_withdrawals = mean_daily_withdrawals
        self.std_daily_withdrawals = std_daily_withdrawals

        self.mean_new_loans = mean_new_loans
        self.std_new_loans = std_new_loans
        self.mean_repayments = mean_repayments
        self.std_repayments = std_repayments
        self.mean_defaults = mean_defaults
        self.std_defaults = std_defaults

        self.stress_manager = stress_manager or StressEventManager(seed=seed)

    def _sample(self, mean: float, std: float) -> float:
        return max(0.0, float(self.rng.normal(mean, std)))

    def simulate_day(self, day_index: int) -> dict:
        modifiers = self.stress_manager.maybe_trigger(day_index, {
            "cash_reserve": self.cash_reserve,
            "total_deposits": self.total_deposits,
            "loan_assets": self.loan_assets,
        })

        daily_deposits = self._sample(self.mean_daily_deposits * modifiers.get("deposits", 1.0),
                                      self.std_daily_deposits)
        daily_withdrawals = self._sample(self.mean_daily_withdrawals * modifiers.get("withdrawals", 1.0),
                                         self.std_daily_withdrawals)

        new_loans = self._sample(self.mean_new_loans * modifiers.get("new_loans", 1.0),
                                 self.std_new_loans)
        loan_repayments = self._sample(self.mean_repayments * modifiers.get("repayments", 1.0),
                                       self.std_repayments)
        loan_defaults = self._sample(self.mean_defaults * modifiers.get("defaults", 1.0),
                                     self.std_defaults)

        # Update aggregates
        self.cash_reserve += daily_deposits
        self.cash_reserve -= daily_withdrawals
        self.cash_reserve -= new_loans
        self.cash_reserve += loan_repayments
        self.cash_reserve -= loan_defaults

        # Loans affect assets and deposits
        self.loan_assets += new_loans
        self.loan_assets -= loan_repayments
        self.loan_assets -= loan_defaults

        # Deposits change by net inflow (deposits - withdrawals)
        self.total_deposits += (daily_deposits - daily_withdrawals)
        if self.total_deposits <= 0:
            self.total_deposits = 1.0

        liquidity_ratio = self.cash_reserve / self.total_deposits

        return {
            "day": int(day_index),
            "cash_reserve": self.cash_reserve,
            "total_deposits": self.total_deposits,
            "loan_assets": self.loan_assets,
            "daily_deposits": daily_deposits,
            "daily_withdrawals": daily_withdrawals,
            "new_loans": new_loans,
            "loan_repayments": loan_repayments,
            "loan_defaults": loan_defaults,
            "liquidity_ratio": liquidity_ratio,
        }

    def simulate_days(self, n_days: int = 365) -> pd.DataFrame:
        rows = []
        for d in range(1, n_days + 1):
            rows.append(self.simulate_day(d))
        return pd.DataFrame(rows)


if __name__ == "__main__":
    sim = Simulator(seed=42)
    df = sim.simulate_days(10)
    print(df.head())
