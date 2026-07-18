"""Investment Portfolio tab — holdings table with weekly death-cross exit flags and
fixed-WACC DCF fair values.

Exposes ``render_portfolio_tab()``, called by app.py.
Holdings source of truth: data/portfolio.csv (edit in the UI for the session; download
and commit the CSV to persist across redeploys).
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import streamlit as st

from dcf.data import compute_default_assumptions, fetch_company_data
from dcf.formatting import fmt_currency, fmt_pct
from dcf.model import Assumptions, run_dcf
from leaps.crosses import death_cross_status, weekly_ema_cross
from leaps.prices import batch_download_closes

_PORTFOLIO_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "portfolio.csv")
PORTFOLIO_WACC = 0.10  # user's fixed convention for portfolio valuation


def _load_portfolio_csv() -> pd.DataFrame:
    try:
        df = pd.read_csv(_PORTFOLIO_CSV)
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        if "name" not in df.columns:
            df["name"] = df["ticker"]
        return df[["ticker", "name"]].dropna(subset=["ticker"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame({"ticker": [], "name": []})


@st.cache_data(ttl=3600, show_spinner=False)
def _dcf_fair_value(ticker: str) -> Optional[float]:
    """Existing DCF engine with a fixed 10% WACC and auto smart defaults."""
    try:
        data = fetch_company_data(ticker)
    except Exception:
        return None
    if not data.revenue or not data.shares_outstanding:
        return None
    d = compute_default_assumptions(data)
    ebitda_margin = (data.ebitda / data.revenue) if (data.ebitda and data.revenue) else 0.0
    a = Assumptions(
        base_revenue=data.revenue,
        projection_years=int(d["projection_years"]),
        stage1_growth=float(d["stage1_growth"]),
        terminal_growth=float(d["terminal_growth"]),
        fcf_margin=float(d["fcf_margin"]),
        tax_rate=float(d["tax_rate"]),
        fade_growth=True,
        beta=float(d["beta"]),
        equity_risk_premium=float(d["equity_risk_premium"]),
        wacc_override=PORTFOLIO_WACC,
        ebitda_margin=ebitda_margin,
        exit_multiple=float(d["exit_multiple"]),
        net_debt=data.net_debt or 0.0,
        shares_outstanding=data.shares_outstanding,
    )
    return run_dcf(a).fair_value_per_share


def _range_position(daily_close: pd.Series) -> Optional[float]:
    """Where the last price sits in its trailing 52-week range (0 = low, 1 = high)."""
    close = daily_close.dropna().iloc[-252:]
    if len(close) < 60:
        return None
    lo, hi = float(close.min()), float(close.max())
    if hi <= lo:
        return None
    return (float(close.iloc[-1]) - lo) / (hi - lo)


_FLAG_ICONS = {"Green": "🟢 Green", "Amber": "🟠 Amber", "Red": "🔴 Red", "—": "—"}


def render_portfolio_tab() -> None:
    st.caption(
        "Your holdings with a **weekly 50/200-EMA death-cross exit flag** (3% buffer) and a "
        f"**DCF fair value at a fixed {PORTFOLIO_WACC:.0%} WACC**. Educational — not investment advice."
    )

    with st.container(border=True):
        st.markdown("#### Holdings")
        st.caption(
            "Edit below for this session. To persist changes, **💾 download** the CSV and commit "
            "it to the repo as `data/portfolio.csv` (or edit the file on GitHub directly)."
        )
        edited = st.data_editor(
            st.session_state.get("pf_editor_seed", _load_portfolio_csv()),
            num_rows="dynamic", key="pf_editor", width="stretch", hide_index=True,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", required=True),
                "name": st.column_config.TextColumn("Name"),
            },
        )
        holdings = edited.copy()
        holdings["ticker"] = holdings["ticker"].astype(str).str.upper().str.strip()
        holdings = holdings[holdings["ticker"] != ""].drop_duplicates("ticker")

        b1, b2 = st.columns(2)
        analyze = b1.button("🔄 Analyze portfolio", type="primary", width="stretch",
                            key="pf_analyze_btn")
        b2.download_button("💾 Download portfolio.csv",
                           holdings.to_csv(index=False).encode("utf-8"),
                           "portfolio.csv", "text/csv", width="stretch", key="pf_download")

    if analyze:
        if holdings.empty:
            st.warning("Add at least one ticker to analyze.")
            return
        tickers = holdings["ticker"].tolist()
        names = dict(zip(holdings["ticker"], holdings["name"].fillna("")))
        bar = st.progress(0.0, text="Downloading weekly history…")

        weekly = batch_download_closes(tickers, period="10y", interval="1wk",
                                       min_bars=30,
                                       progress=lambda f, m: bar.progress(f * 0.3, text=m))
        bar.progress(0.35, text="Downloading daily history…")
        daily = batch_download_closes(tickers, period="1y", interval="1d", min_bars=30,
                                      progress=lambda f, m: bar.progress(0.35 + f * 0.15, text=m))

        rows = []
        for i, t in enumerate(tickers):
            bar.progress(0.5 + 0.5 * i / max(len(tickers), 1), text=f"Valuing {t}…")
            state = weekly_ema_cross(weekly[t]) if t in weekly else None
            flag = death_cross_status(state)
            price = float(daily[t].dropna().iloc[-1]) if t in daily and len(daily[t].dropna()) else None
            rng = _range_position(daily[t]) if t in daily else None
            fv = _dcf_fair_value(t)
            upside = (fv / price - 1) if (fv and price) else None
            verdict = ("Undervalued" if upside is not None and upside > 0.10 else
                       "Overvalued" if upside is not None and upside < -0.10 else
                       "Fairly valued" if upside is not None else "—")
            rows.append({
                "Ticker": t,
                "Name": names.get(t, ""),
                "Price": price,
                "Death cross": flag,
                "Weekly EMA gap %": round(state.gap * 100, 2) if state else None,
                "52W range %": round(rng * 100, 0) if rng is not None else None,
                f"DCF value (WACC {PORTFOLIO_WACC:.0%})": fv,
                "Upside %": round(upside * 100, 1) if upside is not None else None,
                "Verdict": verdict,
            })
        bar.empty()
        st.session_state["pf_results"] = pd.DataFrame(rows)

    results = st.session_state.get("pf_results")
    if results is None:
        st.info("Click **Analyze portfolio** to compute exit flags and DCF values.")
        return
    if results.empty:
        st.warning("No results — check the tickers in your portfolio.")
        return

    dcf_col = f"DCF value (WACC {PORTFOLIO_WACC:.0%})"
    n_red = int((results["Death cross"] == "Red").sum())
    n_amber = int((results["Death cross"] == "Amber").sum())
    n_under = int((results["Verdict"] == "Undervalued").sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Holdings", len(results))
    m2.metric("🔴 Exit signals", n_red)
    m3.metric("🟠 Warning (in buffer)", n_amber)
    m4.metric("Undervalued (DCF)", n_under)

    disp = results.copy()
    disp["Death cross"] = disp["Death cross"].map(lambda v: _FLAG_ICONS.get(v, v))
    disp["Price"] = disp["Price"].map(lambda v: fmt_currency(v) if v is not None else "—")
    disp[dcf_col] = disp[dcf_col].map(lambda v: fmt_currency(v) if v is not None else "—")
    st.dataframe(
        disp, hide_index=True, width="stretch",
        column_config={
            "Weekly EMA gap %": st.column_config.NumberColumn(format="%.2f%%"),
            "52W range %": st.column_config.NumberColumn(format="%.0f%%"),
            "Upside %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    st.caption(
        "**Death cross** (weekly): 🟢 50-week EMA at/above the 200-week EMA · 🟠 crossed below but "
        "within the 3% buffer · 🔴 more than 3% below → exit signal. '—' = under ~4 years of "
        "history (200-week EMA not meaningful). **52W range %**: 0% = at the 52-week low, 100% = "
        "at the high. DCF uses the app's FCFF engine with auto defaults and a fixed "
        f"{PORTFOLIO_WACC:.0%} WACC; verdict bands at ±10%."
    )
