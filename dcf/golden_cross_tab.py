"""Golden Cross Scanner tab — weekly 50/200-EMA golden crosses across S&P 500 + Nasdaq-100.

Exposes ``render_golden_cross_tab()``, called by app.py. Entry-signal mirror of the
portfolio tab's death-cross exit flag: "fresh" crosses sit inside the 3% buffer above the
200-week EMA; "recently crossed" happened within the last ~8 weeks but have run past it.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dcf import nasdaq100, sp500
from dcf.formatting import fmt_currency
from leaps.crosses import golden_cross_class, weekly_ema_cross
from leaps.prices import batch_download_closes

_BUFFER = 0.03
_RECENT_WEEKS = 8
_CARDS_PER_ROW = 3


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _universe() -> pd.DataFrame:
    sp = sp500.load_sp500()[["ticker", "name"]]
    ndx = nasdaq100.load_nasdaq100()[["ticker", "name"]]
    return pd.concat([sp, ndx]).drop_duplicates("ticker").reset_index(drop=True)


def _card(col, row: dict, kind: str) -> None:
    accent = "#0a7d32" if kind == "fresh" else "#8a6d00"
    badge = "FRESH CROSS" if kind == "fresh" else "RECENTLY CROSSED"
    weeks = row["weeks_since_cross"]
    weeks_txt = f"{weeks} wk ago" if weeks is not None else "—"
    col.markdown(
        f"""
<div style="border:1px solid #e0e0e0; border-left:5px solid {accent}; border-radius:10px;
            padding:12px 14px; margin-bottom:12px; background:#fff;
            box-shadow:0 1px 2px rgba(0,0,0,0.04);">
  <div style="font-size:0.72rem; font-weight:700; letter-spacing:0.04em; color:{accent};">{badge}</div>
  <div style="font-size:1.15rem; font-weight:700; margin-top:2px;">{row['ticker']}</div>
  <div style="font-size:0.8rem; color:#666; white-space:nowrap; overflow:hidden;
              text-overflow:ellipsis;">{row['name']}</div>
  <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:0.85rem;">
    <span>Price<br><b>{fmt_currency(row['price'])}</b></span>
    <span>EMA gap<br><b>{row['gap'] * 100:+.2f}%</b></span>
    <span>Crossed<br><b>{weeks_txt}</b></span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_golden_cross_tab() -> None:
    st.caption(
        "Scans **S&P 500 + Nasdaq-100** for weekly **golden crosses** (50-week EMA above the "
        "200-week EMA) — the entry-signal mirror of the portfolio death-cross flag. "
        "**Fresh** = gap still inside the 3% buffer (early-entry zone) · **Recently crossed** = "
        f"bullish cross within the last {_RECENT_WEEKS} weeks, gap already beyond 3%."
    )

    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        c1.caption("~560 unique tickers; weekly history download takes ~1–3 min on a cold run, "
                   "then it's cached.")
        run = c2.button("🔎 Run scan", type="primary", width="stretch", key="gc_run_btn")

    if run:
        uni = _universe()
        names = dict(zip(uni["ticker"], uni["name"]))
        bar = st.progress(0.0, text="Starting…")
        weekly = batch_download_closes(
            uni["ticker"].tolist(), period="10y", interval="1wk", min_bars=30,
            progress=lambda f, m: bar.progress(f * 0.9, text=m), chunk_size=200,
        )
        rows = []
        n_had_data = 0
        for t, series in weekly.items():
            state = weekly_ema_cross(series)
            if state is None:
                continue
            n_had_data += 1
            kind = golden_cross_class(state, _BUFFER, _RECENT_WEEKS)
            if kind:
                rows.append({
                    "ticker": t, "name": names.get(t, t), "price": state.last_close,
                    "gap": state.gap, "weeks_since_cross": state.weeks_since_cross,
                    "class": kind,
                })
        bar.empty()
        st.session_state["gc_results"] = pd.DataFrame(rows)
        st.session_state["gc_counts"] = {
            "universe": len(uni), "downloaded": len(weekly), "enough_history": n_had_data,
        }

    results = st.session_state.get("gc_results")
    counts = st.session_state.get("gc_counts")

    if counts:
        fresh_n = int((results["class"] == "fresh").sum()) if results is not None and not results.empty else 0
        recent_n = int((results["class"] == "recent").sum()) if results is not None and not results.empty else 0
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Universe", counts["universe"])
        k2.metric("Had data", counts["downloaded"])
        k3.metric("≥ 4y history", counts["enough_history"])
        k4.metric("✨ Fresh", fresh_n)
        k5.metric("🌱 Recent", recent_n)

    if results is None:
        st.info("Click **Run scan** to find weekly golden-cross entries.")
        return
    if results.empty:
        st.warning("No fresh or recently-crossed golden crosses right now — the entry zone is "
                   "empty. Re-run after the weekly close, or when market breadth improves.")
        return

    fresh = results[results["class"] == "fresh"].sort_values("gap").reset_index(drop=True)
    recent = results[results["class"] == "recent"].sort_values("weeks_since_cross").reset_index(drop=True)

    if not fresh.empty:
        st.markdown(f"#### ✨ Fresh crosses — in the 3% entry zone ({len(fresh)})")
        for i in range(0, len(fresh), _CARDS_PER_ROW):
            cols = st.columns(_CARDS_PER_ROW)
            for j, (_, row) in enumerate(fresh.iloc[i:i + _CARDS_PER_ROW].iterrows()):
                _card(cols[j], row, "fresh")

    if not recent.empty:
        st.markdown(f"#### 🌱 Recently crossed — past the buffer ({len(recent)})")
        for i in range(0, len(recent), _CARDS_PER_ROW):
            cols = st.columns(_CARDS_PER_ROW)
            for j, (_, row) in enumerate(recent.iloc[i:i + _CARDS_PER_ROW].iterrows()):
                _card(cols[j], row, "recent")

    export = results.copy()
    export["gap %"] = (export.pop("gap") * 100).round(2)
    st.download_button("⬇️ Download CSV", export.to_csv(index=False).encode("utf-8"),
                       "golden_crosses.csv", "text/csv", key="gc_download")
    st.caption("Weekly bars; 50/200-week EMAs. Educational — not investment advice.")
