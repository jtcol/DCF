"""Data-quality checks that help the user judge how much to trust a valuation.

Each check returns a Flag with a severity so the UI can colour-code them. These do NOT
block the valuation — they contextualize it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import CompanyData

# Sectors where unlevered FCFF DCF is unreliable (cash flows are financing-driven).
_FINANCIAL_SECTORS = {"Financials", "Financial Services", "Real Estate"}


@dataclass
class Flag:
    severity: str  # "error" | "warning" | "info" | "ok"
    message: str


def assess(data: CompanyData) -> list[Flag]:
    flags: list[Flag] = []

    # Missing core fields
    if data.missing:
        flags.append(Flag("warning", f"Missing data fields: {', '.join(data.missing)}. "
                                     "Defaults were substituted where possible."))

    # Sector caveat
    if data.sector in _FINANCIAL_SECTORS:
        flags.append(Flag(
            "warning",
            f"{data.sector} companies have financing-driven cash flows; an unlevered FCFF DCF "
            "is a poor fit here. Treat the fair value as indicative only.",
        ))

    # History depth
    n_hist = len(data.hist_revenue.dropna()) if data.hist_revenue is not None else 0
    if n_hist == 0:
        flags.append(Flag("error", "No historical revenue available — defaults are not data-grounded."))
    elif n_hist < 3:
        flags.append(Flag("warning", f"Only {n_hist} year(s) of history available; trend-based "
                                     "defaults (growth, FCF margin) are weak."))
    else:
        flags.append(Flag("ok", f"{n_hist} years of historical financials available."))

    # Negative / volatile FCF
    fcff = data.hist_fcff.dropna() if data.hist_fcff is not None else None
    if fcff is not None and len(fcff) > 0:
        if (fcff < 0).any():
            flags.append(Flag("warning", "One or more historical years had negative free cash flow; "
                                         "the FCF margin assumption may be unstable."))
        if len(fcff) >= 2:
            mean = float(np.mean(fcff))
            std = float(np.std(fcff))
            if mean != 0 and abs(std / mean) > 0.75:
                flags.append(Flag("info", "Free cash flow has been volatile historically "
                                          "(coefficient of variation > 0.75)."))

    # Negative net debt (net cash) — fine, just informative
    if data.net_debt is not None and data.net_debt < 0:
        flags.append(Flag("info", "Company has net cash (negative net debt); this adds to equity value."))

    # Beta sanity
    if data.beta is None:
        flags.append(Flag("info", "Beta unavailable; defaulted to 1.0 for cost of equity."))
    elif data.beta <= 0:
        flags.append(Flag("warning", f"Reported beta ({data.beta:.2f}) is non-positive; cost of "
                                     "equity may be understated."))

    # Effective tax rate sanity
    if data.effective_tax_rate is None:
        flags.append(Flag("info", "Effective tax rate could not be derived; defaulted to 21%."))

    return flags


def severity_rank(flags: list[Flag]) -> str:
    """Overall confidence label derived from the worst flag present."""
    severities = {f.severity for f in flags}
    if "error" in severities:
        return "Low"
    if "warning" in severities:
        return "Moderate"
    return "High"
