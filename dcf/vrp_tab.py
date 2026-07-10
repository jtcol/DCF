"""Volatility Risk Premium tab — rank tickers by option-premium richness (IV vs RV).

Exposes ``render_vrp_tab()``, called by app.py.
"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from dcf import nasdaq100, sp500
from dcf.formatting import fmt_big
from leaps.vrp import VRPParams, run_vrp_scan


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _nasdaq100_df() -> pd.DataFrame:
    return nasdaq100.load_nasdaq100()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _sp500_df() -> pd.DataFrame:
    return sp500.load_sp500()


def _watchlist_df(raw: str) -> pd.DataFrame:
    tickers = list(dict.fromkeys(t.upper() for t in re.split(r"[,\s]+", raw.strip()) if t))
    return pd.DataFrame({"ticker": tickers, "name": tickers, "sector": ""})


def _guidance() -> None:
    with st.expander("📘 What is the Volatility Risk Premium? — read me", expanded=False):
        st.markdown(
            """
The **Volatility Risk Premium (VRP)** is the tendency for **implied volatility (IV)** — what options
*price in* — to exceed the volatility a stock **actually realizes (RV)**. When it does, option
**sellers** are, on average, paid more than the risk they take on.

**What this screen computes per ticker:**
- **IV30** — at-the-money implied vol for ~30 days, from the options chain (interpolated to 30 DTE).
- **RV30** — annualized realized vol over the last ~21 trading days (how much it *actually* moved).
- **VRP = IV30 − RV30**, and the **IV/RV ratio**.

**How to read it:**
- **IV/RV > 1 → "Good (rich)"** — options are pricing a *bigger* move than the stock has been
  making → statistically favorable for **selling** premium (mean-reversion edge). *Ranked highest.*
- **IV/RV < 1 → "Poor (cheap)"** — you'd be paid less than realized movement; better to *buy* or wait.
- **IV-rank proxy > 50 → "Rich"**, **< 30 → "Cheap"** — where current IV sits vs the stock's own
  trailing 1-year volatility range. Confirms whether premium is rich/cheap *relative to itself*.

**Using it:** the best short-premium candidates are **high IV/RV *and* a Rich IV regime**. Prefer
**defined-risk** structures (spreads), mind earnings dates (IV crush), and size positions modestly.

> **IV-rank is a proxy** (vs realized vol) — yfinance has no historical IV. Educational tool, **not
> investment advice**; verify every contract in your broker.
            """
        )


def render_vrp_tab() -> None:
    st.caption(
        "Ranks tickers by how **rich** their option premium is: 30-day implied vol vs 30-day "
        "realized vol. Highest **IV/RV** first — the option-seller's edge."
    )
    _guidance()

    with st.container(border=True):
        st.markdown("#### Scan settings")
        c1, c2 = st.columns([1, 2])
        scope = c1.radio("Universe", ["Nasdaq-100", "S&P 500", "Watchlist"], key="vrp_scope")
        watchlist_raw = ""
        if scope == "Watchlist":
            watchlist_raw = c2.text_input("Tickers (comma/space separated)",
                                          "AAPL, MSFT, KO, NKE, TSLA, NVDA", key="vrp_watch")
        else:
            c2.caption(
                "Every ticker needs an options fetch, so this runs the whole index: "
                "**Nasdaq-100 ≈ 1–3 min**, **S&P 500 ≈ 5–15 min** on a cold run "
                "(faster within the cache window)."
            )
        run = st.button("🔎 Run scan", type="primary", use_container_width=True, key="vrp_run_btn")

    if run:
        if scope == "Nasdaq-100":
            uni = _nasdaq100_df()
        elif scope == "S&P 500":
            uni = _sp500_df()
        else:
            uni = _watchlist_df(watchlist_raw)

        if uni.empty:
            st.warning("No tickers to scan. Add some tickers to your watchlist.")
            return

        bar = st.progress(0.0, text="Starting…")

        def _progress(frac: float, msg: str) -> None:
            bar.progress(frac, text=msg)

        with st.spinner("Scanning…"):
            results, counts = run_vrp_scan(uni, VRPParams(), _progress)
        bar.empty()
        st.session_state["vrp_results"] = results
        st.session_state["vrp_counts"] = counts

    results = st.session_state.get("vrp_results")
    counts = st.session_state.get("vrp_counts")

    if counts is not None:
        k1, k2, k3 = st.columns(3)
        k1.metric("Universe", counts.universe)
        k2.metric("Had price data", counts.had_prices)
        k3.metric("Had options/IV", counts.had_iv)

    if results is None:
        st.info("Choose a universe and click **Run scan**.")
        return
    if results.empty:
        st.warning("No tickers returned usable IV + RV. Options data may be temporarily "
                   "unavailable — try again shortly or use a smaller watchlist.")
        return

    rich = int((results["Premium"] == "Good (rich)").sum())
    st.markdown(f"#### Ranked by IV/RV — {len(results)} tickers ({rich} with rich premium)")

    disp = results.copy()
    disp["Price"] = disp["Price"].map(lambda v: f"${v:,.2f}")
    st.dataframe(
        disp,
        hide_index=True,
        use_container_width=True,
        column_config={
            "IV/RV": st.column_config.NumberColumn(format="%.2f"),
            "IV-rank proxy %": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.caption(
        "**IV/RV > 1** = options richer than realized movement (good for sellers). **IV regime** "
        "from the IV-rank *proxy*: Rich > 50, Cheap < 30. Best short-premium setups = high IV/RV "
        "**and** a Rich regime. IV-rank is a proxy (current IV vs the stock's 1-yr realized-vol range)."
    )
    st.download_button("⬇️ Download CSV", results.to_csv(index=False).encode("utf-8"),
                       "vrp_ranking.csv", "text/csv", key="vrp_download")
