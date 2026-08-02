"""Fetch and normalize company financial data from yfinance, and derive smart defaults.

yfinance is an unofficial scraper of Yahoo Finance: row labels and availability vary by
company and over time. Every extraction here is defensive — missing fields return ``None``
and are recorded so the UI can flag them rather than crash.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# --- Candidate row labels (yfinance uses different names across companies/versions) -------

_REVENUE_LABELS = ["Total Revenue", "Operating Revenue", "Revenue"]
_EBIT_LABELS = ["EBIT", "Operating Income", "Total Operating Income As Reported"]
_EBITDA_LABELS = ["EBITDA", "Normalized EBITDA"]
_PRETAX_LABELS = ["Pretax Income", "Income Before Tax"]
_TAX_LABELS = ["Tax Provision", "Income Tax Expense", "Provision For Income Taxes"]
_INTEREST_LABELS = ["Interest Expense", "Interest Expense Non Operating", "Net Interest Income"]
_DA_LABELS = [
    "Depreciation And Amortization",
    "Depreciation Amortization Depletion",
    "Depreciation",
    "Reconciled Depreciation",
]
_CAPEX_LABELS = ["Capital Expenditure", "Capital Expenditures", "Purchase Of PPE"]
_CHG_WC_LABELS = ["Change In Working Capital", "Changes In Working Capital"]
_FCF_LABELS = ["Free Cash Flow"]
_OCF_LABELS = [
    "Operating Cash Flow",
    "Total Cash From Operating Activities",
    "Cash Flow From Continuing Operating Activities",
    "Cash Flowsfromusedin Operating Activities Direct",
]
_DEBT_LABELS = ["Total Debt"]
_LT_DEBT_LABELS = ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]
_CURR_DEBT_LABELS = ["Current Debt", "Current Debt And Capital Lease Obligation"]
_CASH_LABELS = [
    "Cash And Cash Equivalents",
    "Cash Cash Equivalents And Short Term Investments",
    "Cash Financial",
]
_SHARES_LABELS = ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"]


@dataclass
class CompanyData:
    ticker: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    currency: Optional[str] = None
    current_price: Optional[float] = None
    shares_outstanding: Optional[float] = None
    market_cap: Optional[float] = None
    beta: Optional[float] = None

    # Most-recent statement values (absolute currency units)
    revenue: Optional[float] = None
    ebit: Optional[float] = None
    ebitda: Optional[float] = None
    da: Optional[float] = None
    capex: Optional[float] = None  # stored as a positive outflow magnitude
    change_in_wc: Optional[float] = None
    interest_expense: Optional[float] = None
    pretax_income: Optional[float] = None
    tax_expense: Optional[float] = None

    # Balance sheet
    total_debt: Optional[float] = None
    cash: Optional[float] = None

    # Derived
    net_debt: Optional[float] = None
    effective_tax_rate: Optional[float] = None
    fcf_latest: Optional[float] = None  # reported Free Cash Flow if available

    # Trailing-twelve-month (TTM) free cash flow, summed from the last 4 quarters
    fcf_ttm: Optional[float] = None
    revenue_ttm: Optional[float] = None
    fcf_margin_ttm: Optional[float] = None

    # Best historical FCF margin year (a data-grounded anchor for margin expansion, not an
    # invented bull case) and analyst forward growth consensus (reference only, not auto-applied)
    fcf_margin_peak: Optional[float] = None
    analyst_growth: Optional[float] = None
    analyst_growth_horizon: Optional[str] = None  # e.g. "long-term (5y)" or "next fiscal year"

    # Historical series (index = period end date, most recent first)
    hist_revenue: pd.Series = field(default_factory=pd.Series)
    hist_ebit: pd.Series = field(default_factory=pd.Series)
    hist_fcff: pd.Series = field(default_factory=pd.Series)
    hist_fcf_margin: pd.Series = field(default_factory=pd.Series)

    as_of: Optional[str] = None
    missing: list[str] = field(default_factory=list)
    raw_income: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_cashflow: pd.DataFrame = field(default_factory=pd.DataFrame)
    raw_balance: pd.DataFrame = field(default_factory=pd.DataFrame)


def _find_row(df: pd.DataFrame, labels: list[str]) -> Optional[pd.Series]:
    """Return the first matching row (as a Series indexed by period) for any candidate label."""
    if df is None or df.empty:
        return None
    index_lower = {str(i).strip().lower(): i for i in df.index}
    for label in labels:
        key = label.strip().lower()
        if key in index_lower:
            row = df.loc[index_lower[key]]
            # Some companies have duplicate labels -> DataFrame slice; take first row.
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return row.dropna()
    return None


def _latest(row: Optional[pd.Series]) -> Optional[float]:
    if row is None or len(row) == 0:
        return None
    try:
        return float(row.iloc[0])
    except (ValueError, TypeError):
        return None


def get_risk_free_rate() -> float:
    """10-year Treasury yield from ^TNX (quoted in percent), as a decimal.

    Falls back to FALLBACK_RISK_FREE_RATE from the environment if the fetch fails.
    """
    fallback = float(os.getenv("FALLBACK_RISK_FREE_RATE", "0.042"))
    ticker = os.getenv("RISK_FREE_RATE_TICKER", "^TNX")
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if not hist.empty:
            return float(hist["Close"].dropna().iloc[-1]) / 100.0
    except Exception:
        pass
    return fallback


def _parse_growth_cell(g: pd.DataFrame, row_label: str) -> Optional[float]:
    if row_label not in g.index:
        return None
    row = g.loc[row_label]
    col = "stockTrend" if "stockTrend" in row.index else ("stock" if "stock" in row.index else row.index[0])
    try:
        val = float(row[col])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    # Some yfinance versions report this as a whole-number percent (e.g. 18.2) rather than
    # a decimal (0.182); normalize by magnitude rather than assuming a fixed convention.
    if abs(val) > 3.0:
        val = val / 100.0
    return float(np.clip(val, -0.50, 2.00))


def _fetch_analyst_growth(tk: "yf.Ticker") -> tuple[Optional[float], Optional[str]]:
    """Analyst consensus forward growth for the stock itself. Reference figure only — never
    used to set a default.

    Prefers the long-term (~5y) estimate; Yahoo frequently leaves that field empty for
    individual stocks, so falls back to the next-fiscal-year estimate when it's missing —
    always paired with an accurate horizon label so a 1-year figure is never shown as 5-year.
    Defensive end to end: any missing property, empty frame, or absent row/column returns
    (None, None) rather than guessing.
    """
    try:
        g = tk.growth_estimates
    except Exception:
        return None, None
    if g is None or not isinstance(g, pd.DataFrame) or g.empty:
        return None, None

    ltg = _parse_growth_cell(g, "LTG")
    if ltg is not None:
        return ltg, "long-term (5y) consensus"
    ny = _parse_growth_cell(g, "+1y")
    if ny is not None:
        return ny, "next fiscal year consensus"
    return None, None


def fetch_company_data(ticker: str) -> CompanyData:
    """Pull income statement, cash flow, balance sheet and quote data for ``ticker``."""
    ticker = ticker.strip().upper()
    data = CompanyData(ticker=ticker, as_of=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
    tk = yf.Ticker(ticker)

    # --- Quote / profile -------------------------------------------------------------
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    fast = {}
    try:
        fast = dict(tk.fast_info or {})
    except Exception:
        fast = {}

    data.name = info.get("longName") or info.get("shortName") or ticker
    data.sector = info.get("sector")
    data.industry = info.get("industry")
    data.currency = info.get("currency") or fast.get("currency")
    data.current_price = (
        fast.get("last_price") or info.get("currentPrice") or info.get("regularMarketPrice")
    )
    # Price fallback: last close from history (a lighter endpoint than .info).
    if not data.current_price:
        try:
            h = tk.history(period="5d")
            if not h.empty:
                data.current_price = float(h["Close"].dropna().iloc[-1])
        except Exception:
            pass
    data.shares_outstanding = info.get("sharesOutstanding") or fast.get("shares")
    data.market_cap = info.get("marketCap") or fast.get("market_cap")
    data.beta = info.get("beta")

    # --- Statements (annual) ---------------------------------------------------------
    try:
        income = tk.income_stmt
    except Exception:
        income = pd.DataFrame()
    try:
        cashflow = tk.cashflow
    except Exception:
        cashflow = pd.DataFrame()
    try:
        balance = tk.balance_sheet
    except Exception:
        balance = pd.DataFrame()

    data.raw_income = income if isinstance(income, pd.DataFrame) else pd.DataFrame()
    data.raw_cashflow = cashflow if isinstance(cashflow, pd.DataFrame) else pd.DataFrame()
    data.raw_balance = balance if isinstance(balance, pd.DataFrame) else pd.DataFrame()

    rev_row = _find_row(income, _REVENUE_LABELS)
    ebit_row = _find_row(income, _EBIT_LABELS)
    ebitda_row = _find_row(income, _EBITDA_LABELS)
    da_row = _find_row(cashflow, _DA_LABELS)
    capex_row = _find_row(cashflow, _CAPEX_LABELS)
    chg_wc_row = _find_row(cashflow, _CHG_WC_LABELS)
    fcf_row = _find_row(cashflow, _FCF_LABELS)

    data.revenue = _latest(rev_row)
    data.ebit = _latest(ebit_row)
    data.ebitda = _latest(ebitda_row)
    data.da = _latest(da_row)
    capex_val = _latest(capex_row)
    data.capex = abs(capex_val) if capex_val is not None else None  # store as positive magnitude
    data.change_in_wc = _latest(chg_wc_row)
    data.interest_expense = _latest(_find_row(income, _INTEREST_LABELS))
    data.pretax_income = _latest(_find_row(income, _PRETAX_LABELS))
    data.tax_expense = _latest(_find_row(income, _TAX_LABELS))
    data.fcf_latest = _latest(fcf_row)

    # --- Trailing-twelve-month FCF (sum of last 4 quarters) --------------------------
    try:
        q_cashflow = tk.quarterly_cashflow
    except Exception:
        q_cashflow = pd.DataFrame()
    try:
        q_income = tk.quarterly_income_stmt
    except Exception:
        q_income = pd.DataFrame()
    data.fcf_ttm, data.revenue_ttm, data.fcf_margin_ttm = _compute_ttm_fcf(q_cashflow, q_income)

    # EBITDA fallback = EBIT + D&A
    if data.ebitda is None and data.ebit is not None and data.da is not None:
        data.ebitda = data.ebit + data.da

    # --- Balance sheet ---------------------------------------------------------------
    debt = _latest(_find_row(balance, _DEBT_LABELS))
    if debt is None:
        lt = _latest(_find_row(balance, _LT_DEBT_LABELS)) or 0.0
        cur = _latest(_find_row(balance, _CURR_DEBT_LABELS)) or 0.0
        debt = lt + cur if (lt or cur) else None
    data.total_debt = debt
    data.cash = _latest(_find_row(balance, _CASH_LABELS))
    if data.total_debt is not None:
        data.net_debt = data.total_debt - (data.cash or 0.0)

    # Shares fallback from the balance sheet when .info/.fast_info didn't supply it.
    if not data.shares_outstanding:
        sh = _latest(_find_row(balance, _SHARES_LABELS))
        if sh and sh > 0:
            data.shares_outstanding = sh

    # Market-cap fallback = price × shares (avoids depending on the flaky .info endpoint).
    if not data.market_cap and data.current_price and data.shares_outstanding:
        data.market_cap = data.current_price * data.shares_outstanding

    # --- Effective tax rate ----------------------------------------------------------
    if data.tax_expense is not None and data.pretax_income not in (None, 0):
        rate = data.tax_expense / data.pretax_income
        # Guard against nonsense (loss years, refunds)
        data.effective_tax_rate = float(np.clip(rate, 0.0, 0.50))

    # --- Historical series -----------------------------------------------------------
    data.hist_revenue = rev_row if rev_row is not None else pd.Series(dtype=float)
    data.hist_ebit = ebit_row if ebit_row is not None else pd.Series(dtype=float)
    data.hist_fcff = _historical_fcff(ebit_row, da_row, capex_row, chg_wc_row, data.effective_tax_rate)
    if not data.hist_revenue.empty and not data.hist_fcff.empty:
        aligned = data.hist_fcff.reindex(data.hist_revenue.index)
        data.hist_fcf_margin = (aligned / data.hist_revenue).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data.hist_fcf_margin) >= 2:
        peak = float(data.hist_fcf_margin.max())
        if np.isfinite(peak):
            data.fcf_margin_peak = peak

    data.analyst_growth, data.analyst_growth_horizon = _fetch_analyst_growth(tk)

    # --- Record what is missing ------------------------------------------------------
    for fld, val in [
        ("revenue", data.revenue),
        ("EBIT", data.ebit),
        ("D&A", data.da),
        ("capex", data.capex),
        ("total debt", data.total_debt),
        ("cash", data.cash),
        ("shares outstanding", data.shares_outstanding),
        ("current price", data.current_price),
        ("beta", data.beta),
    ]:
        if val is None:
            data.missing.append(fld)

    return data


def _ttm_sum(df: pd.DataFrame, labels: list[str], n: int = 4) -> Optional[float]:
    """Sum the most recent ``n`` quarterly values of the first matching row."""
    row = _find_row(df, labels)
    if row is None or len(row) == 0:
        return None
    vals = row.iloc[:n].dropna()
    if len(vals) == 0:
        return None
    return float(vals.sum())


def _compute_ttm_fcf(q_cashflow: pd.DataFrame, q_income: pd.DataFrame):
    """Trailing-twelve-month FCF, revenue, and FCF margin from quarterly statements.

    Prefers the reported quarterly "Free Cash Flow"; otherwise reconstructs it as
    Operating Cash Flow + Capex (capex is reported negative). Returns (fcf, revenue, margin).
    """
    fcf = _ttm_sum(q_cashflow, _FCF_LABELS)
    if fcf is None:
        ocf = _ttm_sum(q_cashflow, _OCF_LABELS)
        capex = _ttm_sum(q_cashflow, _CAPEX_LABELS)  # negative outflow
        if ocf is not None:
            fcf = ocf + (capex if capex is not None else 0.0)
    revenue = _ttm_sum(q_income, _REVENUE_LABELS)
    margin = (fcf / revenue) if (fcf is not None and revenue) else None
    return fcf, revenue, margin


def _historical_fcff(ebit_row, da_row, capex_row, chg_wc_row, tax_rate) -> pd.Series:
    """FCFF per period = EBIT*(1-tax) + D&A - Capex - ChangeInWC, aligned on common dates."""
    if ebit_row is None:
        return pd.Series(dtype=float)
    t = tax_rate if tax_rate is not None else 0.21
    idx = ebit_row.index
    ebit = ebit_row.reindex(idx).fillna(0.0)
    da = (da_row.reindex(idx).fillna(0.0) if da_row is not None else pd.Series(0.0, index=idx))
    capex = (capex_row.reindex(idx) if capex_row is not None else pd.Series(0.0, index=idx)).fillna(0.0)
    chg_wc = (chg_wc_row.reindex(idx) if chg_wc_row is not None else pd.Series(0.0, index=idx)).fillna(0.0)
    # capex is reported negative; ChangeInWC sign in yfinance is a cash-flow contribution.
    fcff = ebit * (1 - t) + da + capex + chg_wc
    return fcff.dropna()


def compute_default_assumptions(data: CompanyData) -> dict:
    """Derive sensible starting assumptions from history (all overridable in the UI)."""
    defaults = {
        "projection_years": 5,
        "stage1_growth": 0.08,
        "terminal_growth": 0.025,
        "fcf_margin": 0.12,
        "tax_rate": data.effective_tax_rate if data.effective_tax_rate is not None else 0.21,
        "equity_risk_premium": float(os.getenv("DEFAULT_EQUITY_RISK_PREMIUM", "0.05")),
        "beta": data.beta if data.beta is not None else 1.0,
        "exit_multiple": 12.0,
    }

    # Revenue CAGR from history (oldest -> newest), capped to a sane band.
    rev = data.hist_revenue.dropna()
    if len(rev) >= 2:
        newest, oldest = float(rev.iloc[0]), float(rev.iloc[-1])
        years = len(rev) - 1
        if oldest > 0 and newest > 0 and years > 0:
            cagr = (newest / oldest) ** (1 / years) - 1
            defaults["stage1_growth"] = round(float(np.clip(cagr, -0.05, 0.30)), 4)

    # FCF margin: prefer TTM (last 4 quarters), then historical average, then latest annual.
    if data.fcf_margin_ttm is not None and np.isfinite(data.fcf_margin_ttm):
        defaults["fcf_margin"] = round(float(np.clip(data.fcf_margin_ttm, -0.10, 0.60)), 4)
    elif not data.hist_fcf_margin.empty:
        m = float(data.hist_fcf_margin.mean())
        if np.isfinite(m):
            defaults["fcf_margin"] = round(float(np.clip(m, -0.10, 0.60)), 4)
    elif data.revenue and data.fcf_latest:
        defaults["fcf_margin"] = round(float(np.clip(data.fcf_latest / data.revenue, -0.10, 0.60)), 4)

    # Terminal/target FCF margin: default to the company's own historical peak margin — a
    # data-grounded anchor for margin expansion (e.g. a heavy-capex-now business that has
    # already proven a higher margin in a leaner year), not an invented bull case. Only
    # expands the default (never below the current-margin default); with no clear peak, the
    # terminal margin simply equals the current one (fade is then a no-op).
    defaults["fcf_margin_terminal"] = defaults["fcf_margin"]
    if data.fcf_margin_peak is not None and np.isfinite(data.fcf_margin_peak):
        peak_clipped = float(np.clip(data.fcf_margin_peak, -0.10, 0.60))
        if peak_clipped > defaults["fcf_margin"]:
            defaults["fcf_margin_terminal"] = round(peak_clipped, 4)

    # Analyst forward growth consensus — reference only, shown in the UI but never auto-applied.
    defaults["analyst_growth"] = data.analyst_growth
    defaults["analyst_growth_horizon"] = data.analyst_growth_horizon

    # Exit multiple from current EV/EBITDA if we can estimate EV.
    if data.ebitda and data.ebitda > 0 and data.market_cap:
        ev = data.market_cap + (data.net_debt or 0.0)
        mult = ev / data.ebitda
        if np.isfinite(mult) and mult > 0:
            defaults["exit_multiple"] = round(float(np.clip(mult, 4.0, 30.0)), 1)

    return defaults
