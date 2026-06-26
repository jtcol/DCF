"""FCFF (unlevered) DCF engine: WACC via CAPM, revenue-driven 2-stage projection,
dual terminal value (Gordon Growth + Exit Multiple), and per-share fair value.

All monetary inputs/outputs are in the company's reporting currency (absolute units).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Assumptions:
    # Projection drivers
    base_revenue: float
    projection_years: int
    stage1_growth: float
    terminal_growth: float
    fcf_margin: float
    tax_rate: float
    fade_growth: bool = True  # linearly fade growth from stage1 (yr1) to terminal (yrN)

    # WACC components (CAPM)
    risk_free: float = 0.042
    beta: float = 1.0
    equity_risk_premium: float = 0.05
    cost_of_debt: float = 0.05  # pre-tax
    equity_weight: float = 1.0
    debt_weight: float = 0.0
    wacc_override: Optional[float] = None

    # Terminal value
    ebitda_margin: float = 0.0  # terminal EBITDA = terminal revenue * this margin (for exit multiple)
    exit_multiple: float = 12.0
    use_gordon: bool = True
    use_exit_multiple: bool = True

    # Bridge to equity
    net_debt: float = 0.0
    shares_outstanding: float = 0.0


@dataclass
class DCFResult:
    wacc: float
    cost_of_equity: float
    after_tax_cost_of_debt: float
    projection: pd.DataFrame  # year, growth, revenue, fcff, discount_factor, pv_fcff
    pv_fcff_total: float
    tv_gordon: Optional[float]
    tv_exit: Optional[float]
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float
    equity_value: float
    fair_value_per_share: Optional[float]
    terminal_pv_weight: float  # fraction of EV coming from the terminal value (a risk signal)
    warnings: list[str]


def compute_wacc(a: Assumptions) -> tuple[float, float, float]:
    """Return (wacc, cost_of_equity, after_tax_cost_of_debt)."""
    cost_of_equity = a.risk_free + a.beta * a.equity_risk_premium
    after_tax_cost_of_debt = a.cost_of_debt * (1 - a.tax_rate)
    if a.wacc_override is not None:
        return a.wacc_override, cost_of_equity, after_tax_cost_of_debt
    # Normalize weights defensively.
    we, wd = a.equity_weight, a.debt_weight
    total = we + wd
    if total <= 0:
        we, wd = 1.0, 0.0
    else:
        we, wd = we / total, wd / total
    wacc = we * cost_of_equity + wd * after_tax_cost_of_debt
    return wacc, cost_of_equity, after_tax_cost_of_debt


def _growth_path(a: Assumptions) -> list[float]:
    """Per-year revenue growth rates for years 1..N."""
    n = a.projection_years
    if not a.fade_growth or n <= 1:
        return [a.stage1_growth] * n
    # Linear fade from stage1 (year 1) to terminal (year N).
    return list(np.linspace(a.stage1_growth, a.terminal_growth, n))


def run_dcf(a: Assumptions) -> DCFResult:
    warnings: list[str] = []
    wacc, coe, atcod = compute_wacc(a)

    if wacc <= a.terminal_growth:
        warnings.append(
            f"WACC ({wacc:.1%}) is at or below terminal growth ({a.terminal_growth:.1%}); "
            "Gordon Growth terminal value is invalid and was disabled."
        )

    growths = _growth_path(a)
    rows = []
    revenue = a.base_revenue
    for year in range(1, a.projection_years + 1):
        g = growths[year - 1]
        revenue = revenue * (1 + g)
        fcff = revenue * a.fcf_margin
        df = 1.0 / ((1 + wacc) ** year)
        rows.append(
            {
                "Year": year,
                "Growth": g,
                "Revenue": revenue,
                "FCFF": fcff,
                "Discount Factor": df,
                "PV of FCFF": fcff * df,
            }
        )
    projection = pd.DataFrame(rows)
    pv_fcff_total = float(projection["PV of FCFF"].sum())

    final_revenue = float(projection["Revenue"].iloc[-1])
    final_fcff = float(projection["FCFF"].iloc[-1])
    n = a.projection_years
    discount_n = 1.0 / ((1 + wacc) ** n)

    # --- Terminal value: Gordon Growth ----------------------------------------------
    tv_gordon = None
    if a.use_gordon and wacc > a.terminal_growth:
        fcff_next = final_fcff * (1 + a.terminal_growth)
        tv_gordon = fcff_next / (wacc - a.terminal_growth)

    # --- Terminal value: Exit Multiple ----------------------------------------------
    tv_exit = None
    if a.use_exit_multiple and a.ebitda_margin > 0:
        terminal_ebitda = final_revenue * a.ebitda_margin
        tv_exit = terminal_ebitda * a.exit_multiple
    elif a.use_exit_multiple and a.ebitda_margin <= 0:
        warnings.append("Exit-multiple terminal value skipped: EBITDA margin unavailable/non-positive.")

    # --- Combine selected terminal values -------------------------------------------
    tvs = [tv for tv in (tv_gordon, tv_exit) if tv is not None]
    if tvs:
        terminal_value = float(np.mean(tvs))
    else:
        terminal_value = 0.0
        warnings.append("No valid terminal value could be computed; fair value reflects explicit FCFF only.")

    pv_terminal_value = terminal_value * discount_n
    enterprise_value = pv_fcff_total + pv_terminal_value
    equity_value = enterprise_value - a.net_debt

    fair_value_per_share = None
    if a.shares_outstanding and a.shares_outstanding > 0:
        fair_value_per_share = equity_value / a.shares_outstanding

    terminal_pv_weight = (pv_terminal_value / enterprise_value) if enterprise_value > 0 else 0.0
    if terminal_pv_weight > 0.80:
        warnings.append(
            f"Terminal value drives {terminal_pv_weight:.0%} of enterprise value — the result is "
            "highly sensitive to terminal assumptions (WACC, terminal growth, exit multiple)."
        )

    return DCFResult(
        wacc=wacc,
        cost_of_equity=coe,
        after_tax_cost_of_debt=atcod,
        projection=projection,
        pv_fcff_total=pv_fcff_total,
        tv_gordon=tv_gordon,
        tv_exit=tv_exit,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal_value,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        fair_value_per_share=fair_value_per_share,
        terminal_pv_weight=terminal_pv_weight,
        warnings=warnings,
    )
