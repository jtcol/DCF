"""Quarterly-baseline dual DCF (Operating Cash Flow & Free Cash Flow).

Methodology (user-specified):
1. Ingest up to 6 quarters of real reported cash flow.
2. Strip quarters that are 2+ standard-deviation outliers.
3. Average the survivors into a clean quarterly baseline; annualize (x4).
4. Project the annual baseline forward N years at a defined growth rate.
5. Discount every future year at WACC.
6. Sum the discounted years (+ Gordon Growth terminal value) -> fair value per share.
7. Run the whole sequence twice: once on OCF, once on FCF.

Pure logic — no Streamlit imports — so the math is unit-testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .data import _CAPEX_LABELS, _FCF_LABELS, _OCF_LABELS, _find_row

MAX_QUARTERS = 6
OUTLIER_Z = 2.0
MIN_QUARTERS_FOR_STRIP = 4  # with fewer points, sigma is meaningless — keep all


@dataclass
class QuarterlyCF:
    ticker: str
    ocf: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    fcf: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    fcf_source: str = ""  # "reported" | "ocf-capex" | ""


@dataclass
class QDCFResult:
    stream: str                       # "OCF" | "FCF"
    quarters: pd.Series               # all ingested quarters (newest first)
    kept: pd.Series
    dropped: pd.Series
    baseline_quarterly: float
    baseline_annual: float
    projection: pd.DataFrame          # Year, Cash Flow, Discount Factor, PV
    pv_years_total: float             # spec-strict: sum of discounted explicit years
    terminal_value: Optional[float]
    pv_terminal_value: float
    fair_value_with_tv: Optional[float]
    fair_value_years_only: Optional[float]  # the conservative "floor" (no TV)
    warnings: list[str] = field(default_factory=list)


def fetch_quarterly_cf(ticker: str) -> QuarterlyCF:
    """Pull up to MAX_QUARTERS of reported quarterly OCF and FCF (newest first)."""
    out = QuarterlyCF(ticker=ticker.strip().upper())
    try:
        qcf = yf.Ticker(out.ticker).quarterly_cashflow
    except Exception:
        qcf = pd.DataFrame()
    if not isinstance(qcf, pd.DataFrame) or qcf.empty:
        return out

    ocf_row = _find_row(qcf, _OCF_LABELS)
    if ocf_row is not None:
        out.ocf = ocf_row.dropna().iloc[:MAX_QUARTERS].astype(float)

    fcf_row = _find_row(qcf, _FCF_LABELS)
    if fcf_row is not None and len(fcf_row.dropna()) > 0:
        out.fcf = fcf_row.dropna().iloc[:MAX_QUARTERS].astype(float)
        out.fcf_source = "reported"
    elif ocf_row is not None:
        capex_row = _find_row(qcf, _CAPEX_LABELS)  # reported negative
        if capex_row is not None:
            fcf = (ocf_row + capex_row.reindex(ocf_row.index).fillna(0.0)).dropna()
            out.fcf = fcf.iloc[:MAX_QUARTERS].astype(float)
            out.fcf_source = "ocf-capex"
    return out


def strip_outliers(s: pd.Series, z: float = OUTLIER_Z) -> tuple[pd.Series, pd.Series]:
    """Drop values >= z standard deviations from the mean. Returns (kept, dropped)."""
    s = s.dropna()
    if len(s) < MIN_QUARTERS_FOR_STRIP:
        return s, s.iloc[0:0]
    mean, sigma = float(s.mean()), float(s.std(ddof=0))
    if sigma == 0 or not np.isfinite(sigma):
        return s, s.iloc[0:0]
    zscores = (s - mean).abs() / sigma
    kept, dropped = s[zscores < z], s[zscores >= z]
    if len(kept) == 0:  # pathological: never drop everything
        return s, s.iloc[0:0]
    return kept, dropped


def run_quarterly_dcf(
    stream: str,
    quarters: pd.Series,
    growth: float,
    wacc: float,
    terminal_growth: float,
    years: int,
    shares_outstanding: float,
) -> Optional[QDCFResult]:
    """Run the clean->baseline->project->discount->sum sequence on one cash-flow stream."""
    quarters = quarters.dropna()
    if len(quarters) == 0:
        return None

    kept, dropped = strip_outliers(quarters)
    baseline_q = float(kept.mean())
    baseline_a = baseline_q * 4.0

    warnings: list[str] = []
    if len(quarters) < MIN_QUARTERS_FOR_STRIP:
        warnings.append(f"Only {len(quarters)} quarters available — too few for outlier "
                        "screening, all quarters kept.")
    if baseline_a <= 0:
        warnings.append("Clean baseline cash flow is non-positive; fair value is not meaningful.")

    rows = []
    cf = baseline_a
    for year in range(1, years + 1):
        cf = cf * (1 + growth)
        df_ = 1.0 / ((1 + wacc) ** year)
        rows.append({"Year": year, "Cash Flow": cf, "Discount Factor": df_, "PV": cf * df_})
    projection = pd.DataFrame(rows)
    pv_years = float(projection["PV"].sum())

    tv = None
    pv_tv = 0.0
    if wacc > terminal_growth and baseline_a > 0:
        final_cf = float(projection["Cash Flow"].iloc[-1])
        tv = final_cf * (1 + terminal_growth) / (wacc - terminal_growth)
        pv_tv = tv / ((1 + wacc) ** years)
    elif wacc <= terminal_growth:
        warnings.append(f"WACC ({wacc:.1%}) is at or below terminal growth "
                        f"({terminal_growth:.1%}); terminal value disabled.")

    fv_with_tv = fv_years_only = None
    if shares_outstanding and shares_outstanding > 0:
        fv_with_tv = (pv_years + pv_tv) / shares_outstanding
        fv_years_only = pv_years / shares_outstanding
    else:
        warnings.append("Shares outstanding unavailable — cannot compute per-share value.")

    return QDCFResult(
        stream=stream, quarters=quarters, kept=kept, dropped=dropped,
        baseline_quarterly=baseline_q, baseline_annual=baseline_a,
        projection=projection, pv_years_total=pv_years,
        terminal_value=tv, pv_terminal_value=pv_tv,
        fair_value_with_tv=fv_with_tv, fair_value_years_only=fv_years_only,
        warnings=warnings,
    )


def run_dual(
    qcf: QuarterlyCF,
    growth: float,
    wacc: float,
    terminal_growth: float,
    years: int,
    shares_outstanding: float,
) -> dict[str, Optional[QDCFResult]]:
    """Run the full sequence on both streams. Keys: 'OCF', 'FCF'."""
    return {
        "OCF": run_quarterly_dcf("OCF", qcf.ocf, growth, wacc, terminal_growth,
                                 years, shares_outstanding),
        "FCF": run_quarterly_dcf("FCF", qcf.fcf, growth, wacc, terminal_growth,
                                 years, shares_outstanding),
    }
