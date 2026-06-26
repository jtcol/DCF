"""Reverse DCF: solve for the stage-1 revenue growth rate that the current market price implies.

Holds every other assumption fixed and uses bisection (no SciPy dependency) to find the
growth rate where the model's fair value per share equals the current market price.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

from .model import Assumptions, run_dcf


@dataclass
class ReverseDCFResult:
    implied_growth: Optional[float]
    converged: bool
    message: str


def _fair_value_at_growth(base: Assumptions, growth: float) -> Optional[float]:
    a = copy.copy(base)
    a.stage1_growth = growth
    return run_dcf(a).fair_value_per_share


def solve_implied_growth(
    base: Assumptions,
    market_price: float,
    low: float = -0.30,
    high: float = 0.60,
    tol: float = 1e-4,
    max_iter: int = 100,
) -> ReverseDCFResult:
    """Find stage-1 growth g such that fair_value(g) == market_price via bisection.

    Fair value is monotonically increasing in growth, so bisection is reliable.
    """
    if market_price is None or market_price <= 0:
        return ReverseDCFResult(None, False, "No valid market price to solve against.")
    if base.shares_outstanding in (None, 0):
        return ReverseDCFResult(None, False, "Shares outstanding unavailable; cannot reverse-solve.")

    f_low = _fair_value_at_growth(base, low)
    f_high = _fair_value_at_growth(base, high)
    if f_low is None or f_high is None:
        return ReverseDCFResult(None, False, "Model did not return a value at the search bounds.")

    # Market price must lie within the achievable fair-value range.
    if market_price < f_low:
        return ReverseDCFResult(
            low, False,
            f"Implied growth is below the {low:.0%} search floor — the market price is lower than "
            "this model can produce even with deeply negative growth.",
        )
    if market_price > f_high:
        return ReverseDCFResult(
            high, False,
            f"Implied growth exceeds the {high:.0%} search ceiling — the market is pricing in growth "
            "above what this model bounds.",
        )

    lo, hi = low, high
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        fv = _fair_value_at_growth(base, mid)
        if fv is None:
            return ReverseDCFResult(None, False, "Model returned no value during the search.")
        if abs(fv - market_price) < tol * max(market_price, 1.0):
            return ReverseDCFResult(mid, True, "Converged.")
        if fv < market_price:
            lo = mid
        else:
            hi = mid
    return ReverseDCFResult((lo + hi) / 2, True, "Converged (max iterations).")
