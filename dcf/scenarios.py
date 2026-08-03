"""Bear / Base / Bull scenario builder for the DCF engine.

Pure logic, no Streamlit. The user's editable inputs define the **Base** case; Bear and Bull are
derived from it by documented, data-grounded rules — never invented from scratch and never tuned
to land near the market price. All three scenarios share the SAME discount rate (WACC): scenarios
flex cash-flow assumptions, not risk, which is standard practice (risk is already reflected in the
cash-flow spread across scenarios, so discounting bear/bull differently would double-count it).

Guardrail constants below are named and commented so the "attunement" is auditable, not a black box.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .model import Assumptions

# --- Guardrail constants (industry-standard sanity bands) --------------------------------
BEAR_GROWTH_MIN, BEAR_GROWTH_MAX = -0.15, 0.15   # bear ≠ collapse; floor stops runaway pessimism
BULL_GROWTH_MIN, BULL_GROWTH_MAX = -0.15, 0.40   # bull capped well below hyper-growth extremes
BEAR_TERMINAL_GROWTH_CAP = 0.020                  # bear terminal growth: at/below ~long-run real GDP
BULL_TERMINAL_GROWTH_CAP = 0.030                  # bull terminal growth: still below long-run nominal GDP
BULL_TERMINAL_GROWTH_UPLIFT = 1.20                # bull = 20% relative bump on the user's own terminal growth
MARGIN_FLOOR, MARGIN_CAP = -0.10, 0.60            # same band as the existing FCF-margin inputs
BULL_MARGIN_TTM_UPLIFT = 1.15                     # bull margin ceiling = 15% above the TTM margin
BEAR_EXIT_MULT_FACTOR, BULL_EXIT_MULT_FACTOR = 0.8, 1.2
EXIT_MULT_FLOOR, EXIT_MULT_CAP = 4.0, 30.0
BULL_HOLD_YEARS = 3                               # bull "compounding phase" before growth fades
BULL_EXTRA_PROJECTION_YEARS = 2                   # extra runway so the post-hold fade isn't a cliff
FALLBACK_BEAR_GROWTH_FACTOR = 0.5                 # used only when no analyst low estimate exists
FALLBACK_BULL_GROWTH_FACTOR = 1.5                 # used only when no analyst high/avg estimate exists

DEFAULT_WEIGHTS = {"bear": 0.25, "base": 0.50, "bull": 0.25}


@dataclass
class ScenarioAnchors:
    """Data anchors pulled from CompanyData / compute_default_assumptions that ground the
    Bear/Bull rules in real analyst dispersion and company history, not arbitrary multipliers."""
    analyst_growth_avg: Optional[float] = None
    analyst_growth_low: Optional[float] = None
    analyst_growth_high: Optional[float] = None
    hist_fcf_margin_mean: Optional[float] = None
    hist_fcf_margin_peak: Optional[float] = None
    fcf_margin_ttm: Optional[float] = None


def _bear_growth(base_growth: float, analyst_low: Optional[float]) -> float:
    candidate = analyst_low if analyst_low is not None else base_growth * FALLBACK_BEAR_GROWTH_FACTOR
    return float(np.clip(min(base_growth, candidate), BEAR_GROWTH_MIN, BEAR_GROWTH_MAX))


def _bull_growth(base_growth: float, analyst_high: Optional[float], analyst_avg: Optional[float]) -> float:
    candidates = [base_growth]
    candidates.append(analyst_high if analyst_high is not None else base_growth * FALLBACK_BULL_GROWTH_FACTOR)
    if analyst_avg is not None:
        candidates.append(analyst_avg)
    return float(np.clip(max(candidates), BULL_GROWTH_MIN, BULL_GROWTH_MAX))


def _bear_terminal_margin(base_current_margin: float, hist_mean_margin: Optional[float]) -> float:
    candidate = hist_mean_margin if hist_mean_margin is not None else base_current_margin
    return float(np.clip(min(base_current_margin, candidate), MARGIN_FLOOR, MARGIN_CAP))


def _bull_terminal_margin(
    base_terminal_margin: float, hist_peak_margin: Optional[float], ttm_margin: Optional[float]
) -> float:
    candidates = [base_terminal_margin]
    if hist_peak_margin is not None:
        candidates.append(hist_peak_margin)
    if ttm_margin is not None:
        candidates.append(ttm_margin * BULL_MARGIN_TTM_UPLIFT)
    return float(np.clip(max(candidates), MARGIN_FLOOR, MARGIN_CAP))


def build_scenarios(base: Assumptions, anchors: ScenarioAnchors) -> dict[str, Assumptions]:
    """Derive Bear and Bull ``Assumptions`` from the Base case. Same WACC-driving fields
    (risk_free, beta, equity_risk_premium, cost_of_debt, weights, wacc_override) are inherited
    unchanged from ``base`` in every scenario — only cash-flow assumptions flex.
    """
    bear_growth = _bear_growth(base.stage1_growth, anchors.analyst_growth_low)
    bull_growth = _bull_growth(base.stage1_growth, anchors.analyst_growth_high, anchors.analyst_growth_avg)

    bear_terminal_growth = float(min(base.terminal_growth, BEAR_TERMINAL_GROWTH_CAP))
    bull_terminal_growth = float(min(base.terminal_growth * BULL_TERMINAL_GROWTH_UPLIFT, BULL_TERMINAL_GROWTH_CAP))

    base_terminal_margin = base.fcf_margin_terminal if base.fcf_margin_terminal is not None else base.fcf_margin
    bear_terminal_margin = _bear_terminal_margin(base.fcf_margin, anchors.hist_fcf_margin_mean)
    bull_terminal_margin = _bull_terminal_margin(
        base_terminal_margin, anchors.hist_fcf_margin_peak, anchors.fcf_margin_ttm
    )

    bear_exit_mult = float(np.clip(base.exit_multiple * BEAR_EXIT_MULT_FACTOR, EXIT_MULT_FLOOR, EXIT_MULT_CAP))
    bull_exit_mult = float(np.clip(base.exit_multiple * BULL_EXIT_MULT_FACTOR, EXIT_MULT_FLOOR, EXIT_MULT_CAP))

    bear = replace(
        base,
        stage1_growth=bear_growth,
        terminal_growth=bear_terminal_growth,
        fcf_margin_terminal=bear_terminal_margin,
        fade_growth=True,
        hold_years=0,
        exit_multiple=bear_exit_mult,
    )
    bull = replace(
        base,
        stage1_growth=bull_growth,
        terminal_growth=bull_terminal_growth,
        fcf_margin_terminal=bull_terminal_margin,
        fade_growth=True,
        hold_years=BULL_HOLD_YEARS,
        projection_years=base.projection_years + BULL_EXTRA_PROJECTION_YEARS,
        exit_multiple=bull_exit_mult,
    )
    return {"bear": bear, "base": base, "bull": bull}


def weighted_value(fair_values: dict[str, Optional[float]], weights: dict[str, float]) -> Optional[float]:
    """Probability-weighted expected fair value across scenarios.

    Weights need not sum to 1 (normalized here). Scenarios with no fair value (e.g. missing
    shares outstanding) are excluded and the remaining weights renormalized rather than treated
    as zero value.
    """
    pairs = [(fv, weights.get(k, 0.0)) for k, fv in fair_values.items() if fv is not None]
    total_w = sum(w for _, w in pairs)
    if not pairs or total_w <= 0:
        return None
    return sum(fv * w for fv, w in pairs) / total_w
