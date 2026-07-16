"""DCF Valuation tab — the full FCFF discounted-cash-flow dashboard.

Exposes ``render_dcf_tab()``, called by app.py. Inputs live inside the tab (not the
sidebar) so they don't bleed across the app's other top-level tabs.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dcf import sp500
from dcf.data import CompanyData, compute_default_assumptions, fetch_company_data, get_risk_free_rate
from dcf.formatting import fmt_big, fmt_currency, fmt_pct
from dcf.model import Assumptions, run_dcf
from dcf.quality import assess, severity_rank
from dcf.quarterly_dcf import QuarterlyCF, fetch_quarterly_cf, run_dual
from dcf.reverse_dcf import solve_implied_growth


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def _load_universe() -> pd.DataFrame:
    return sp500.load_sp500()


@st.cache_data(show_spinner=True, ttl=3600)
def _load_company(ticker: str) -> CompanyData:
    return fetch_company_data(ticker)


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def _risk_free() -> float:
    return get_risk_free_rate()


@st.cache_data(show_spinner=False, ttl=3600)
def _load_quarterly_cf(ticker: str) -> QuarterlyCF:
    return fetch_quarterly_cf(ticker)


def _severity_icon(sev: str) -> str:
    return {"error": "🔴", "warning": "🟠", "info": "🔵", "ok": "🟢"}.get(sev, "•")


def render_dcf_tab() -> None:
    st.caption(
        "Unlevered (FCFF) discounted cash flow model. Data pulled live from Yahoo Finance via "
        "yfinance. Educational decision-support tool — not investment advice."
    )
    universe = _load_universe()

    # -------------------------------------------------------------------------------
    # INPUTS (kept inside the tab)
    # -------------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("#### 1 · Select company")
        sc1, sc2 = st.columns(2)
        options = sp500.ticker_options(universe)
        default_idx = next((i for i, o in enumerate(options) if o.startswith("AAPL ")), 0)
        picked = sc1.selectbox("S&P 500 constituent", options, index=default_idx)
        free_text = sc2.text_input("…or enter any ticker", value="").strip().upper()
        ticker = free_text if free_text else sp500.parse_ticker_option(picked)

    # Reset cached inputs when the ticker changes so widgets re-init from fresh defaults.
    if st.session_state.get("loaded_ticker") != ticker:
        for k in list(st.session_state.keys()):
            if str(k).startswith("inp_"):
                del st.session_state[k]
        st.session_state["loaded_ticker"] = ticker
        st.session_state["result"] = None

    # Fetch data (cached). A failure returns from the tab (not st.stop) so other tabs survive.
    try:
        data = _load_company(ticker)
    except Exception as exc:  # pragma: no cover - network dependent
        st.error(f"Could not fetch data for **{ticker}**: {exc}")
        return

    if not data.revenue:
        st.warning(
            f"**{ticker}** returned no usable revenue data from yfinance. It may be an invalid ticker, "
            "delisted, or temporarily rate-limited. Try another ticker or retry shortly."
        )

    defaults = compute_default_assumptions(data)
    rf_default = _risk_free()

    mkt_equity = data.market_cap or 0.0
    debt = data.total_debt or 0.0
    denom = mkt_equity + debt
    equity_w_default = (mkt_equity / denom) if denom > 0 else 1.0
    debt_w_default = (debt / denom) if denom > 0 else 0.0

    cost_of_debt_default = 0.05
    if data.interest_expense and data.total_debt:
        cod = abs(data.interest_expense) / data.total_debt
        if 0 < cod < 0.25:
            cost_of_debt_default = round(cod, 4)

    ebitda_margin = (data.ebitda / data.revenue) if (data.ebitda and data.revenue) else 0.0

    with st.container(border=True):
        hc1, hc2 = st.columns([4, 1])
        hc1.markdown("#### 2 · Key assumptions")
        hc1.caption("Pre-filled from data — override as needed.")
        if hc2.button("↩︎ Reset", help="Reset all assumptions to the auto-derived values",
                      width="stretch"):
            for k in list(st.session_state.keys()):
                if str(k).startswith("inp_"):
                    del st.session_state[k]
            st.rerun()

        a1, a2, a3 = st.columns(3)
        projection_years = a1.slider("Projection years", 3, 15, int(defaults["projection_years"]),
                                     key="inp_years")
        stage1_growth = a2.number_input("Stage-1 revenue growth (%)", -20.0, 60.0,
                                        float(defaults["stage1_growth"] * 100), 0.5, key="inp_g1") / 100
        terminal_growth = a3.number_input("Terminal growth (%)", 0.0, 6.0,
                                          float(defaults["terminal_growth"] * 100), 0.1, key="inp_tg") / 100
        b1, b2, b3 = st.columns(3)
        fcf_margin = b1.number_input("FCF margin (% of revenue)", -10.0, 60.0,
                                     float(defaults["fcf_margin"] * 100), 0.5, key="inp_fcfm") / 100
        tax_rate = b2.number_input("Tax rate (%)", 0.0, 50.0,
                                   float(defaults["tax_rate"] * 100), 0.5, key="inp_tax") / 100
        fade_growth = b3.checkbox("Fade growth to terminal rate", value=True, key="inp_fade")

        # Discount rate: single overridable WACC field pre-filled by CAPM.
        risk_free = rf_default
        beta = defaults["beta"]
        erp = defaults["equity_risk_premium"]
        cost_of_debt = cost_of_debt_default
        equity_w = equity_w_default
        debt_w = debt_w_default
        auto_coe = risk_free + beta * erp
        auto_atcod = cost_of_debt * (1 - defaults["tax_rate"])
        _tw = equity_w + debt_w
        _ew, _dw = (equity_w / _tw, debt_w / _tw) if _tw > 0 else (1.0, 0.0)
        auto_wacc = _ew * auto_coe + _dw * auto_atcod

        w1, w2, w3 = st.columns(3)
        wacc_pct = w1.number_input(
            "Discount rate / WACC (%)", 1.0, 30.0, round(auto_wacc * 100, 2), 0.1, key="inp_wacc",
            help="Pre-filled with an automatic CAPM-based estimate. Type your own value to override it.",
        )
        wacc_override = wacc_pct / 100
        use_gordon = w2.checkbox("Gordon Growth TV", value=True, key="inp_gordon")
        use_exit = w2.checkbox("Exit-multiple TV", value=True, key="inp_exit")
        exit_multiple = w3.number_input("Exit EV/EBITDA multiple", 2.0, 40.0,
                                        float(defaults["exit_multiple"]), 0.5, key="inp_mult")
        st.caption(
            f"Auto WACC ≈ {auto_wacc:.1%} (CAPM: rf {risk_free:.1%} + β {beta:.2f} × ERP {erp:.1%}). "
            + (f"Exit multiple auto {defaults['exit_multiple']:.1f}x from current EV/EBITDA."
               if (data.ebitda and data.market_cap)
               else "EV/EBITDA unavailable — exit multiple defaulted to 12.0x.")
        )

        generate = st.button("🚀 Generate Valuation", type="primary", width="stretch")

    assumptions = Assumptions(
        base_revenue=data.revenue or 0.0,
        projection_years=projection_years,
        stage1_growth=stage1_growth,
        terminal_growth=terminal_growth,
        fcf_margin=fcf_margin,
        tax_rate=tax_rate,
        fade_growth=fade_growth,
        risk_free=risk_free,
        beta=beta,
        equity_risk_premium=erp,
        cost_of_debt=cost_of_debt,
        equity_weight=equity_w,
        debt_weight=debt_w,
        wacc_override=wacc_override,
        ebitda_margin=ebitda_margin,
        exit_multiple=exit_multiple,
        use_gordon=use_gordon,
        use_exit_multiple=use_exit,
        net_debt=data.net_debt or 0.0,
        shares_outstanding=data.shares_outstanding or 0.0,
    )

    if generate:
        if not data.revenue:
            st.error("Cannot run a valuation without revenue data.")
        else:
            st.session_state["result"] = run_dcf(assumptions)
            st.session_state["reverse"] = solve_implied_growth(assumptions, data.current_price or 0.0)
            st.session_state["assumptions"] = assumptions
            qcf = _load_quarterly_cf(ticker)
            st.session_state["qdcf"] = run_dual(
                qcf, growth=stage1_growth, wacc=wacc_override,
                terminal_growth=terminal_growth, years=projection_years,
                shares_outstanding=data.shares_outstanding or 0.0,
            )
            st.session_state["qdcf_source"] = qcf.fcf_source

    # -------------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------------
    ccy = data.currency or "USD"
    st.subheader(f"{data.name or ticker}  ·  {ticker}")
    meta = " · ".join([x for x in [data.sector, data.industry, f"Data as of {data.as_of}"] if x])
    st.markdown(f"<span class='small-note'>{meta}</span>", unsafe_allow_html=True)

    with st.expander("ℹ️ What each input field means"):
        glossary = pd.DataFrame(
            [
                ("Projection years", "Length of the explicit forecast period (Stage 1) before a terminal "
                                     "value is applied. Longer horizons rely more on uncertain assumptions."),
                ("Stage-1 revenue growth", "Annual revenue growth rate for the first forecast year. "
                                           "Auto-filled from the company's historical revenue CAGR."),
                ("Fade growth to terminal rate", "If on, growth declines linearly each year from the "
                                                 "stage-1 rate to the terminal rate (more realistic for maturing firms)."),
                ("Terminal growth", "Perpetual growth rate of cash flows after the forecast period. Should be "
                                    "modest — at or below long-run GDP/inflation (typically ~2–3%)."),
                ("FCF margin (% of revenue)", "Free cash flow as a % of revenue. Auto-filled from the company's "
                                              "trailing-twelve-month (TTM) FCF ÷ TTM revenue."),
                ("Tax rate", "Effective corporate tax rate used to unlever EBIT into FCFF. Auto-derived from "
                             "the latest income statement."),
                ("WACC (discount rate)", "Weighted Average Cost of Capital — the rate used to discount future "
                                         "cash flows to today. Pre-filled via CAPM; edit to override."),
                ("Gordon Growth perpetuity", "Terminal value method: assumes FCFF grows forever at the terminal "
                                             "rate. TV = FCFF₍ₙ₊₁₎ ÷ (WACC − terminal growth)."),
                ("Exit multiple (EV/EBITDA)", "Terminal value method: applies an EV/EBITDA multiple to the final "
                                              "year's EBITDA. The app averages the methods you enable."),
                ("Exit EV/EBITDA multiple", "The multiple used by the exit-multiple method. Auto-filled from the "
                                            "company's current implied EV/EBITDA."),
            ],
            columns=["Input field", "What it means"],
        )
        st.table(glossary.set_index("Input field"))

    result = st.session_state.get("result")

    tab_val, tab_proj, tab_qdcf, tab_rev, tab_hist, tab_q = st.tabs(
        ["💰 Valuation", "📊 Projections", "🧮 Quarterly DCF", "🔄 Reverse DCF",
         "📜 Historical & Charts", "✅ Data Quality"]
    )

    with tab_val:
        _render_valuation(result, data, ccy)
    with tab_proj:
        _render_projections(result, ccy)
    with tab_qdcf:
        _render_quarterly(data, ccy)
    with tab_rev:
        _render_reverse(result)
    with tab_hist:
        _render_historical(data)
    with tab_q:
        _render_quality(data, ebitda_margin, ccy)

    st.markdown(
        "<hr><span class='small-note'>Data: Yahoo Finance via yfinance (unofficial; may be delayed or "
        "incomplete). This tool is for education and research only and is not investment advice.</span>",
        unsafe_allow_html=True,
    )


def _render_valuation(result, data, ccy) -> None:
    if result is None:
        st.info("Set your assumptions above and click **Generate Valuation**.")
        return
    fv = result.fair_value_per_share
    price = data.current_price
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fair value / share", fmt_currency(fv, ccy))
    c2.metric("Current price", fmt_currency(price, ccy))
    if fv and price:
        upside = fv / price - 1
        verdict = "Undervalued" if upside > 0.10 else "Overvalued" if upside < -0.10 else "Fairly valued"
        cls = ("verdict-under" if upside > 0.10 else
               "verdict-over" if upside < -0.10 else "verdict-fair")
        c3.metric("Upside / downside", fmt_pct(upside))
        c4.markdown(f"**Verdict**<br><span class='{cls}'>{verdict}</span>", unsafe_allow_html=True)

    for w in result.warnings:
        st.warning(w)

    st.markdown("#### Enterprise → Equity value bridge")
    bridge = pd.DataFrame(
        {
            "Component": ["PV of explicit FCFF", "PV of terminal value", "Enterprise value",
                          "Less: net debt", "Equity value"],
            "Value": [result.pv_fcff_total, result.pv_terminal_value, result.enterprise_value,
                      -(data.net_debt or 0.0), result.equity_value],
        }
    )
    bridge["Value"] = bridge["Value"].map(lambda v: fmt_big(v, ccy))
    st.table(bridge.set_index("Component"))

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("WACC", fmt_pct(result.wacc))
    d2.metric("Cost of equity", fmt_pct(result.cost_of_equity))
    d3.metric("Terminal value % of EV", fmt_pct(result.terminal_pv_weight, 0))
    d4.metric("Shares outstanding", fmt_big(data.shares_outstanding, ""))

    st.markdown("#### Terminal value methods")
    tv = pd.DataFrame(
        {
            "Method": ["Gordon Growth", "Exit Multiple", "Used (average of selected)"],
            "Terminal value": [
                fmt_big(result.tv_gordon, ccy) if result.tv_gordon else "—",
                fmt_big(result.tv_exit, ccy) if result.tv_exit else "—",
                fmt_big(result.terminal_value, ccy),
            ],
        }
    )
    st.table(tv.set_index("Method"))


def _render_projections(result, ccy) -> None:
    if result is None:
        st.info("Generate a valuation to see the year-by-year projection.")
        return
    df = result.projection.copy()
    disp = pd.DataFrame(
        {
            "Year": df["Year"].astype(int),
            "Growth": df["Growth"].map(lambda v: fmt_pct(v)),
            "Revenue": df["Revenue"].map(lambda v: fmt_big(v, ccy)),
            "FCFF": df["FCFF"].map(lambda v: fmt_big(v, ccy)),
            "Discount factor": df["Discount Factor"].map(lambda v: f"{v:.3f}"),
            "PV of FCFF": df["PV of FCFF"].map(lambda v: fmt_big(v, ccy)),
        }
    )
    st.dataframe(disp, hide_index=True, width="stretch")

    n_years = len(df)
    fig_proj = go.Figure()
    fig_proj.add_bar(x=df["Year"], y=df["Revenue"], name="Projected revenue", marker_color="#c6dbef")
    fig_proj.add_scatter(x=df["Year"], y=df["FCFF"], name="Projected FCFF", mode="lines+markers",
                         line=dict(color="#08519c", width=3), yaxis="y2")
    fig_proj.update_layout(
        title=f"{n_years}-Year projection — revenue & free cash flow",
        xaxis=dict(title="Year", dtick=1), yaxis=dict(title=f"Revenue ({ccy})"),
        yaxis2=dict(title=f"FCFF ({ccy})", overlaying="y", side="right", showgrid=False),
        height=400, legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_proj, width="stretch")

    fig = go.Figure()
    fig.add_bar(x=df["Year"], y=df["FCFF"], name="FCFF")
    fig.add_bar(x=df["Year"], y=df["PV of FCFF"], name="PV of FCFF")
    fig.update_layout(barmode="group", title="Projected FCFF vs present value",
                      xaxis=dict(title="Year", dtick=1), yaxis_title=f"{ccy}", height=380,
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")


def _render_quarterly(data, ccy) -> None:
    st.markdown(
        "**Quarterly-baseline DCF** — values the stock from its *actually reported* quarterly "
        "cash flows: ingest up to 6 quarters → strip 2σ outliers → average into a clean "
        "baseline (×4 to annualize) → project at your growth rate → discount at your WACC → "
        "fair value/share. Run twice: **Operating Cash Flow** and **Free Cash Flow**."
    )
    qdcf = st.session_state.get("qdcf")
    if qdcf is None:
        st.info("Click **Generate Valuation** to run the quarterly-baseline DCF "
                "(it uses the same growth, WACC and terminal-growth inputs above).")
        return

    fcf_source = st.session_state.get("qdcf_source", "")
    if fcf_source == "ocf-capex":
        st.caption("FCF derived as Operating Cash Flow − Capex (no reported quarterly FCF row).")

    price = data.current_price
    cols = st.columns(2)
    for col, key, label in zip(cols, ("OCF", "FCF"),
                               ("Operating Cash Flow", "Free Cash Flow")):
        r = qdcf.get(key)
        with col:
            st.markdown(f"#### {label}")
            if r is None:
                st.warning(f"No quarterly {key} data available for this ticker.")
                continue
            fv = r.fair_value_with_tv
            st.metric(f"Fair value / share ({key})", fmt_currency(fv, ccy))
            if fv and price:
                upside = fv / price - 1
                st.metric("vs current price", fmt_pct(upside))
            st.metric("Floor (explicit years only, no TV)",
                      fmt_currency(r.fair_value_years_only, ccy))
            st.caption(
                f"Baseline: {fmt_big(r.baseline_quarterly, ccy)}/qtr → "
                f"{fmt_big(r.baseline_annual, ccy)}/yr · "
                f"{len(r.kept)} of {len(r.quarters)} quarters kept"
            )
            for w in r.warnings:
                st.warning(w)

    st.markdown("#### Data cleaning audit — reported quarters")
    audit_rows = []
    for key in ("OCF", "FCF"):
        r = qdcf.get(key)
        if r is None:
            continue
        dropped_idx = set(r.dropped.index)
        for period, val in r.quarters.items():
            audit_rows.append({
                "Stream": key,
                "Quarter": str(getattr(period, "date", lambda: period)()),
                "Cash flow": fmt_big(float(val), ccy),
                "Status": "❌ dropped (≥2σ outlier)" if period in dropped_idx else "✅ kept",
            })
    if audit_rows:
        st.dataframe(pd.DataFrame(audit_rows), hide_index=True, width="stretch")

    with st.expander("Year-by-year projections"):
        for key in ("OCF", "FCF"):
            r = qdcf.get(key)
            if r is None:
                continue
            st.markdown(f"**{key}**")
            disp = pd.DataFrame({
                "Year": r.projection["Year"].astype(int),
                "Cash flow": r.projection["Cash Flow"].map(lambda v: fmt_big(v, ccy)),
                "Discount factor": r.projection["Discount Factor"].map(lambda v: f"{v:.3f}"),
                "PV": r.projection["PV"].map(lambda v: fmt_big(v, ccy)),
            })
            st.dataframe(disp, hide_index=True, width="stretch")
            st.caption(f"PV of years: {fmt_big(r.pv_years_total, ccy)} · "
                       f"PV of terminal value: {fmt_big(r.pv_terminal_value, ccy)}")

    st.caption(
        "Notes: OCF ignores capex, so its value runs structurally higher than FCF — the gap "
        "reflects capex intensity. Only ~5–6 quarters are available, so the 2σ screen is "
        "indicative (with <4 quarters nothing is stripped). Discounting these levered cash "
        "flows at WACC follows the specified method but mixes firm/equity conventions — treat "
        "results as decision support, not investment advice."
    )


def _render_reverse(result) -> None:
    if result is None:
        st.info("Generate a valuation to run the reverse DCF.")
        return
    reverse = st.session_state.get("reverse")
    st.markdown(
        "The **reverse DCF** holds all of your assumptions fixed except stage-1 growth, then "
        "solves for the growth rate the current market price implies."
    )
    if reverse and reverse.implied_growth is not None:
        implied = reverse.implied_growth
        your_g = st.session_state["assumptions"].stage1_growth
        r1, r2, r3 = st.columns(3)
        r1.metric("Market-implied stage-1 growth", fmt_pct(implied))
        r2.metric("Your stage-1 growth", fmt_pct(your_g))
        r3.metric("Gap (yours − implied)", fmt_pct(your_g - implied))
        if not reverse.converged:
            st.warning(reverse.message)
        elif your_g > implied + 0.01:
            st.success(
                "Your growth assumption is **higher** than the market's — you'd need to be more "
                "optimistic than the market for the stock to be a buy at these other assumptions."
            )
        elif your_g < implied - 0.01:
            st.info(
                "The market is pricing in **higher** growth than you assume — by your assumptions "
                "the market looks optimistic."
            )
        else:
            st.info("Your growth assumption is roughly in line with what the market is pricing in.")
    else:
        st.warning(reverse.message if reverse else "Reverse DCF unavailable.")


def _render_historical(data) -> None:
    rev = data.hist_revenue.dropna() if data.hist_revenue is not None else pd.Series(dtype=float)
    if rev.empty:
        st.info("No historical financial statements available for this ticker.")
        return
    years = [getattr(idx, "year", idx) for idx in rev.index]
    fcff = data.hist_fcff.reindex(rev.index) if data.hist_fcff is not None else pd.Series(dtype=float)

    fig1 = go.Figure()
    fig1.add_bar(x=years, y=rev.values, name="Revenue")
    if fcff is not None and not fcff.dropna().empty:
        fig1.add_bar(x=years, y=fcff.values, name="FCFF")
    fig1.update_layout(barmode="group", title="Revenue & free cash flow (historical)",
                       height=360, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig1, width="stretch")

    margin = data.hist_fcf_margin.dropna() if data.hist_fcf_margin is not None else pd.Series(dtype=float)
    if not margin.empty:
        m_years = [getattr(idx, "year", idx) for idx in margin.index]
        fig2 = go.Figure()
        fig2.add_scatter(x=m_years, y=(margin.values * 100), mode="lines+markers", name="FCF margin %")
        fig2.update_layout(title="FCF margin (%) over time", height=320, yaxis_title="%")
        st.plotly_chart(fig2, width="stretch")

    st.markdown("#### Reported income statement (raw)")
    if not data.raw_income.empty:
        st.dataframe(data.raw_income, width="stretch")
    st.markdown("#### Reported cash flow statement (raw)")
    if not data.raw_cashflow.empty:
        st.dataframe(data.raw_cashflow, width="stretch")


def _render_quality(data, ebitda_margin, ccy) -> None:
    flags = assess(data)
    st.metric("Overall input confidence", severity_rank(flags))
    st.caption(
        "These checks describe how trustworthy the *inputs* are. They do not validate the market's "
        "view — only how well-grounded the model's assumptions can be."
    )
    for f in flags:
        st.markdown(f"{_severity_icon(f.severity)} {f.message}")

    st.markdown("#### Inputs used (auto-derived unless you overrode them)")
    src = pd.DataFrame(
        {
            "Input": ["Base revenue", "Net debt", "Shares outstanding", "Beta", "EBITDA margin",
                      "Current price", "Currency"],
            "Value": [
                fmt_big(data.revenue, ccy), fmt_big(data.net_debt, ccy),
                fmt_big(data.shares_outstanding, ""), f"{data.beta:.2f}" if data.beta else "—",
                fmt_pct(ebitda_margin), fmt_currency(data.current_price, ccy), ccy,
            ],
        }
    )
    st.table(src.set_index("Input"))
