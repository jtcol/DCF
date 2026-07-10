"""LEAPS Screener tab — scan the US market for oversold, bullishly-stacked, low-IV names.

Exposes ``render_leaps_tab()``, called by app.py.
"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from dcf import sp500
from dcf.data import get_risk_free_rate
from dcf.formatting import fmt_big
from leaps import universe as leaps_universe
from leaps.screener import ScanParams, run_leaps_scan


@st.cache_data(ttl=24 * 3600, show_spinner="Loading US market universe…")
def _cached_full_universe() -> pd.DataFrame:
    return leaps_universe.load_universe()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cached_rate() -> float:
    return get_risk_free_rate()


def _watchlist_df(raw: str) -> pd.DataFrame:
    tickers = [t.upper() for t in re.split(r"[,\s]+", raw.strip()) if t]
    tickers = list(dict.fromkeys(tickers))
    return pd.DataFrame({"ticker": tickers, "name": tickers, "sector": "",
                         "market_cap": 0.0, "volume": 0.0, "price": 0.0})


def _sp500_df() -> pd.DataFrame:
    df = sp500.load_sp500().copy()
    for col, val in [("market_cap", 0.0), ("volume", 0.0), ("price", 0.0)]:
        df[col] = val
    return df[["ticker", "name", "sector", "market_cap", "volume", "price"]]


def _guidance() -> None:
    with st.expander("📘 How to pick a great LEAPS — read me", expanded=False):
        st.markdown(
            """
**LEAPS** = options expiring **12+ months** out. Used well, a LEAPS call is a lower-cost,
leveraged **stock replacement**. This screen looks for *quality names on sale*: an oversold
pullback **inside** an intact uptrend, with **cheap** option premium.

**What the three filters mean & why they help:**
- **Weekly RSI < 25** — deeply oversold on a slow timeframe → you're buying a dip, not chasing.
- **Ripster EMA cloud stacked (fast > med > slow)** — the longer-term daily trend is still **up**,
  so the dip is (hopefully) a pullback, not a breakdown. Oversold *in an uptrend* is the sweet spot.
- **ATM IV < 30%** — long options are priced off volatility; **low IV = cheaper premium** and less
  to lose to an IV crush. (See the IV-percentile *proxy* column for cheapness vs the stock's own range.)

**When choosing the actual contract:**
1. **Go deep in-the-money (~0.75–0.85 delta).** It behaves like the stock with less premium at risk
   and far less **theta** (time decay) than at-the-money LEAPS. The suggested contract targets ~0.80Δ.
2. **Give it time** — ≥ 12 months, ideally longer, so a thesis has room to play out.
3. **Demand liquidity** — tight bid/ask **spread** and healthy **open interest**, or you bleed on entry/exit.
4. **Mind earnings** — avoid buying right before a report early in the trade (IV crush / gap risk).
5. **Size it** — total premium should be a small, defined % of the account; LEAPS can still go to zero.

*Educational tool — not investment advice. Verify every contract in your broker before trading.*
            """
        )


def render_leaps_tab() -> None:
    st.caption(
        "Scans the US market with a cheapest-first funnel: market-cap/volume → **weekly RSI < 25** → "
        "**Ripster EMA cloud bullish stack** → **ATM IV < 30%**, then ranks survivors for LEAPS calls."
    )
    _guidance()

    with st.container(border=True):
        st.markdown("#### Scan settings")
        c1, c2, c3 = st.columns(3)
        scope = c1.radio("Universe", ["Full US market", "S&P 500", "Watchlist"], key="lp_scope")
        rsi_threshold = c2.number_input("Weekly RSI below", 5.0, 60.0, 25.0, 1.0, key="lp_rsi")
        iv_threshold = c3.number_input("ATM IV below (%)", 5.0, 100.0, 30.0, 1.0, key="lp_iv") / 100

        watchlist_raw = ""
        if scope == "Watchlist":
            watchlist_raw = st.text_input("Tickers (comma/space separated)",
                                          "AAPL, MSFT, KO, NKE, PEP", key="lp_watch")

        with st.expander("Advanced filters & performance", expanded=False):
            f1, f2, f3 = st.columns(3)
            min_cap_b = f1.number_input("Min market cap ($B)", 0.0, 500.0, 2.0, 0.5, key="lp_cap")
            min_vol_m = f2.number_input("Min avg volume (M shares)", 0.0, 50.0, 1.0, 0.5, key="lp_vol")
            min_price = f3.number_input("Min price ($)", 0.0, 100.0, 5.0, 1.0, key="lp_price")
            g1, g2 = st.columns(2)
            target_delta = g1.slider("Suggested LEAPS delta", 0.55, 0.95, 0.80, 0.05, key="lp_delta")
            limit = g2.number_input("Limit universe (0 = no limit, for speed)", 0, 6000, 0, 50,
                                    key="lp_limit")
            st.caption("A full US-market scan can take several minutes on the first run (bulk price "
                       "download); it's near-instant when re-run within the cache window. Use a limit "
                       "or the Watchlist scope to test quickly.")

        run = st.button("🔎 Run scan", type="primary", width="stretch", key="leaps_run_btn")

    if run:
        if scope == "Full US market":
            uni = leaps_universe.apply_filters(_cached_full_universe(), min_cap_b * 1e9,
                                               min_vol_m * 1e6, min_price)
        elif scope == "S&P 500":
            uni = _sp500_df()
        else:
            uni = _watchlist_df(watchlist_raw)

        if uni.empty:
            st.warning("No tickers to scan. Check your universe/watchlist and filters.")
            return

        params = ScanParams(
            rsi_threshold=rsi_threshold, iv_threshold=iv_threshold,
            min_market_cap=min_cap_b * 1e9, min_volume=min_vol_m * 1e6, min_price=min_price,
            target_delta=target_delta, max_tickers=(int(limit) or None),
        )

        bar = st.progress(0.0, text="Starting…")

        def _progress(frac: float, msg: str) -> None:
            bar.progress(frac, text=msg)

        with st.spinner("Scanning…"):
            results, counts = run_leaps_scan(uni, _cached_rate(), params, _progress)
        bar.empty()

        st.session_state["leaps_results"] = results
        st.session_state["leaps_counts"] = counts

    results = st.session_state.get("leaps_results")
    counts = st.session_state.get("leaps_counts")

    if counts is not None:
        st.markdown("#### Funnel")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Universe", counts.universe)
        k2.metric("Had price data", counts.after_price_data)
        k3.metric(f"RSI < {int(rsi_threshold)}", counts.after_rsi)
        k4.metric("EMA stacked", counts.after_ema)
        k5.metric(f"IV < {int(iv_threshold * 100)}%", counts.after_iv)

    if results is None:
        st.info("Set your filters and click **Run scan**.")
        return
    if results.empty:
        st.warning(
            "No tickers passed all three filters. Weekly RSI < 25 **and** a bullish EMA stack is a "
            "rare combination — try loosening the RSI threshold (e.g. 30–35) or widening IV."
        )
        return

    st.markdown(f"#### Candidates ({len(results)}) — ranked by LEAPS score")
    disp = results.copy()
    disp["Mkt Cap"] = disp["Mkt Cap"].map(lambda v: fmt_big(v) if pd.notna(v) and v > 0 else "—")
    disp["Price"] = disp["Price"].map(lambda v: f"${v:,.2f}")
    disp["LEAPS strike"] = disp["LEAPS strike"].map(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
    st.dataframe(disp, hide_index=True, width="stretch")
    st.caption(
        "LEAPS score (0–100) blends: how oversold (RSI), EMA-stack separation, low ATM IV, low IV "
        "percentile proxy, option liquidity (spread & open interest), and distance from earnings. "
        "IV percentile is a **proxy** (current IV vs the stock's 1-yr realized-vol range) — yfinance "
        "has no historical IV."
    )
    st.download_button("⬇️ Download CSV", results.to_csv(index=False).encode("utf-8"),
                       "leaps_candidates.csv", "text/csv", key="leaps_download")
