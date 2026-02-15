import numpy as np
from typing import Dict, Any, Optional


class StressEventManager:
    """Simple manager to randomly trigger stress modifiers.

    Returns a dict of multipliers for keys: 'withdrawals','defaults','deposits','new_loans','repayments'.
    Uses a local RNG (np.random.Generator) for reproducibility.
    """

    def __init__(self,
                 prob_withdrawal_spike: float = 0.02,
                 prob_default_spike: float = 0.01,
                 prob_deposit_drop: float = 0.02,
                 seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.prob_withdrawal_spike = prob_withdrawal_spike
        self.prob_default_spike = prob_default_spike
        self.prob_deposit_drop = prob_deposit_drop

    def maybe_trigger(self, day_index: int, state: Dict[str, Any]) -> Dict[str, float]:
        modifiers: Dict[str, float] = {}

        # Withdrawal spike: large multiplier
        if self.rng.random() < self.prob_withdrawal_spike:
            modifiers['withdrawals'] = float(self.rng.uniform(1.5, 3.0))

        # Default spike: increases defaults
        if self.rng.random() < self.prob_default_spike:
            modifiers['defaults'] = float(self.rng.uniform(2.0, 5.0))

        # Deposit drop: reduce deposits
        if self.rng.random() < self.prob_deposit_drop:
            modifiers['deposits'] = float(self.rng.uniform(0.2, 0.7))

        return modifiers


if __name__ == "__main__":
    m = StressEventManager(seed=1)
    for d in range(1, 21):
        mods = m.maybe_trigger(d, {})
        if mods:
            print(d, mods)
