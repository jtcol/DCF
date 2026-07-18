"""Multi-tab equity toolkit — DCF valuation + LEAPS screener.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from dcf.dcf_tab import render_dcf_tab
from dcf.golden_cross_tab import render_golden_cross_tab
from dcf.leaps_tab import render_leaps_tab
from dcf.portfolio_tab import render_portfolio_tab
from dcf.vrp_tab import render_vrp_tab

load_dotenv()

st.set_page_config(page_title="Equity Toolkit — DCF & LEAPS", page_icon="📈", layout="wide")

# --- Light professional styling (global) --------------------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem;}
      div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e6e6e6;
          border-radius: 10px; padding: 14px 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);}
      .verdict-under {color: #0a7d32; font-weight: 700;}
      .verdict-over {color: #b00020; font-weight: 700;}
      .verdict-fair {color: #8a6d00; font-weight: 700;}
      .small-note {color: #666; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Equity Toolkit")

tab_pf, tab_dcf, tab_leaps, tab_vrp, tab_gc = st.tabs(
    ["💼 Investment Portfolio", "📈 DCF Valuation", "🎯 LEAPS Screener",
     "🌊 Volatility Risk Premium", "✨ Golden Cross Scanner"]
)

with tab_pf:
    render_portfolio_tab()

with tab_dcf:
    render_dcf_tab()

with tab_leaps:
    render_leaps_tab()

with tab_vrp:
    render_vrp_tab()

with tab_gc:
    render_golden_cross_tab()
