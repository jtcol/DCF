"""Load the current S&P 500 constituent list.

Primary source: Wikipedia's "List of S&P 500 companies" table (live, kept current
by the community). If the live fetch fails (no network, rate limiting, layout change),
fall back to a bundled static CSV so the app still works offline.
"""

from __future__ import annotations

import io
import os
from functools import lru_cache

import pandas as pd
import requests

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
_FALLBACK_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sp500_fallback.csv")


def _normalize_ticker(ticker: str) -> str:
    """yfinance uses '-' where some sources use '.' for share classes (e.g. BRK.B -> BRK-B)."""
    return str(ticker).strip().upper().replace(".", "-")


def _load_from_wikipedia() -> pd.DataFrame:
    # Fetch with a browser User-Agent — Wikipedia 403s pandas/urllib's default agent.
    resp = requests.get(WIKIPEDIA_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    # Columns are typically: Symbol, Security, GICS Sector, GICS Sub-Industry, ...
    df = df.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
    df = df[["ticker", "name", "sector"]].copy()
    df["ticker"] = df["ticker"].map(_normalize_ticker)
    return df.dropna(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)


def _load_from_fallback() -> pd.DataFrame:
    df = pd.read_csv(_FALLBACK_CSV)
    df["ticker"] = df["ticker"].map(_normalize_ticker)
    if "sector" not in df.columns:
        df["sector"] = ""
    return df[["ticker", "name", "sector"]].sort_values("ticker").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_sp500(use_live: bool = True) -> pd.DataFrame:
    """Return a DataFrame with columns: ticker, name, sector.

    Tries the live Wikipedia list first; on any failure, returns the bundled fallback.
    Cached for the process lifetime (Streamlit also wraps this with @st.cache_data).
    """
    if use_live:
        try:
            df = _load_from_wikipedia()
            if len(df) >= 400:  # sanity check that we got a real list
                return df
        except Exception:
            pass
    return _load_from_fallback()


def ticker_options(df: pd.DataFrame) -> list[str]:
    """Build display strings like 'AAPL - Apple Inc.' for a selectbox."""
    return [f"{row.ticker} - {row.name}" for row in df.itertuples(index=False)]


def parse_ticker_option(option: str) -> str:
    """Extract the bare ticker from a 'TICKER - Name' display string."""
    return option.split(" - ", 1)[0].strip().upper()
