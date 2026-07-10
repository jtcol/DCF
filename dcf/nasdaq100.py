"""Load the current Nasdaq-100 constituent list.

Primary source: Wikipedia's "Nasdaq-100" components table. Falls back to a bundled static
CSV if the live fetch fails. Mirrors dcf/sp500.py.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
_FALLBACK_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "nasdaq100_fallback.csv")


def _normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().replace(".", "-")


def _load_from_wikipedia() -> pd.DataFrame:
    tables = pd.read_html(WIKIPEDIA_URL)
    # Find the components table: it has a Ticker/Symbol column and a Company column.
    for tbl in tables:
        cols = {str(c).strip().lower(): c for c in tbl.columns}
        tkr = next((cols[c] for c in ("ticker", "symbol") if c in cols), None)
        name = next((cols[c] for c in ("company", "security") if c in cols), None)
        if tkr is not None and name is not None and len(tbl) >= 90:
            df = tbl.rename(columns={tkr: "ticker", name: "name"})[["ticker", "name"]].copy()
            df["sector"] = ""
            df["ticker"] = df["ticker"].map(_normalize_ticker)
            return df.dropna(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)
    raise ValueError("Nasdaq-100 components table not found on Wikipedia page")


def _load_from_fallback() -> pd.DataFrame:
    df = pd.read_csv(_FALLBACK_CSV)
    df["ticker"] = df["ticker"].map(_normalize_ticker)
    if "sector" not in df.columns:
        df["sector"] = ""
    return df[["ticker", "name", "sector"]].sort_values("ticker").reset_index(drop=True)


@lru_cache(maxsize=1)
def load_nasdaq100(use_live: bool = True) -> pd.DataFrame:
    """Return a DataFrame with columns: ticker, name, sector."""
    if use_live:
        try:
            df = _load_from_wikipedia()
            if len(df) >= 90:
                return df
        except Exception:
            pass
    return _load_from_fallback()
