"""Load a US-stock universe with market cap & average volume attached in one shot.

Primary source: the NASDAQ stock-screener JSON API, which returns ~7,000 US-listed names
with market cap and volume already populated — so the "cap > $2B, volume > 1M" pre-filter
needs no per-ticker download. Falls back to the bundled S&P 500 list if the API is blocked.
"""

from __future__ import annotations

import os
import re

import pandas as pd
import requests

NASDAQ_URL = (
    "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&download=true"
)
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
_FALLBACK_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sp500_fallback.csv")
_VALID_TICKER = re.compile(r"^[A-Z]{1,5}$")


def _parse_money(text) -> float:
    """'$2,345,678,900' or '12.34B' -> float dollars; blanks -> 0."""
    if text is None:
        return 0.0
    s = str(text).strip().replace("$", "").replace(",", "")
    if not s or s in {"--", "N/A", "NA"}:
        return 0.0
    mult = 1.0
    if s[-1] in "KMBT":
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[s[-1]]
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return 0.0


def _fetch_nasdaq() -> pd.DataFrame:
    resp = requests.get(NASDAQ_URL, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    rows = resp.json()["data"]["table"]["rows"]
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "ticker": df["symbol"].str.upper().str.strip(),
            "name": df["name"],
            "sector": df.get("sector", ""),
            "market_cap": df["marketCap"].map(_parse_money),
            "volume": pd.to_numeric(df.get("volume", 0).astype(str).str.replace(",", ""),
                                    errors="coerce").fillna(0.0),
            "price": df["lastsale"].map(_parse_money),
        }
    )
    # Drop warrants, units, preferred, and non-plain tickers (e.g. "ABC.W", "XYZ^A").
    out = out[out["ticker"].map(lambda t: bool(_VALID_TICKER.match(t)))]
    return out.dropna(subset=["ticker"]).drop_duplicates("ticker").reset_index(drop=True)


def _fallback() -> pd.DataFrame:
    df = pd.read_csv(_FALLBACK_CSV)
    df = df.rename(columns={})
    df["ticker"] = df["ticker"].str.upper().str.replace(".", "-", regex=False)
    df["market_cap"] = 0.0  # unknown offline; filters below will treat 0 as "unknown, keep"
    df["volume"] = 0.0
    df["price"] = 0.0
    return df[["ticker", "name", "sector", "market_cap", "volume", "price"]]


def load_universe() -> pd.DataFrame:
    """Full US universe with cap/volume/price columns. Cached by the caller via st.cache_data."""
    try:
        df = _fetch_nasdaq()
        if len(df) >= 1000:
            return df
    except Exception:
        pass
    return _fallback()


def apply_filters(
    df: pd.DataFrame,
    min_market_cap: float = 2e9,
    min_volume: float = 1e6,
    min_price: float = 5.0,
) -> pd.DataFrame:
    """Stage-1 filter. Rows with cap/volume == 0 (unknown, e.g. offline fallback) are kept."""
    m = df.copy()
    cap_ok = (m["market_cap"] >= min_market_cap) | (m["market_cap"] <= 0)
    vol_ok = (m["volume"] >= min_volume) | (m["volume"] <= 0)
    price_ok = (m["price"] >= min_price) | (m["price"] <= 0)
    return m[cap_ok & vol_ok & price_ok].reset_index(drop=True)
