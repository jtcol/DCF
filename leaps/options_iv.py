"""Options / implied-volatility helpers for the LEAPS screen.

- ``atm_iv``: at-the-money IV from the nearest expiry >= min_dte (the screen's hard filter).
- ``realized_vol_percentile``: a *proxy* for IV Rank/Percentile — where current IV sits within
  the trailing 1-year distribution of 20-day realized volatility. (yfinance has no historical
  IV, so a true IV Rank isn't available; this is clearly a proxy.)
- ``suggest_leaps``: pick a ~12-month+ expiry and the ITM call closest to a target delta,
  using a Black-Scholes delta (yfinance option chains don't include greeks).
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_delta(spot: float, strike: float, t_years: float, rate: float, sigma: float) -> float:
    """Black-Scholes call delta = N(d1). Guards against degenerate inputs."""
    if spot <= 0 or strike <= 0 or t_years <= 0 or sigma <= 0:
        return float("nan")
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t_years) / (sigma * math.sqrt(t_years))
    return _norm_cdf(d1)


def _dte(expiry: str) -> int:
    return (datetime.strptime(expiry, "%Y-%m-%d").date() - date.today()).days


def _atm_row(chain: pd.DataFrame, spot: float) -> Optional[pd.Series]:
    if chain is None or chain.empty or "strike" not in chain:
        return None
    idx = (chain["strike"] - spot).abs().idxmin()
    return chain.loc[idx]


def atm_iv(ticker: str, spot: float, min_dte: int = 30) -> Optional[dict]:
    """ATM implied vol (avg of call & put) from the nearest expiry with DTE >= min_dte."""
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options
    except Exception:
        return None
    if not expiries:
        return None
    expiry = next((e for e in expiries if _dte(e) >= min_dte), None)
    if expiry is None:
        return None
    try:
        chain = tk.option_chain(expiry)
    except Exception:
        return None
    call = _atm_row(chain.calls, spot)
    put = _atm_row(chain.puts, spot)
    ivs = [float(x["impliedVolatility"]) for x in (call, put)
           if x is not None and pd.notna(x.get("impliedVolatility")) and x["impliedVolatility"] > 0]
    if not ivs:
        return None
    return {"iv": float(np.mean(ivs)), "expiry": expiry, "dte": _dte(expiry)}


def realized_vol_percentile(daily_close: pd.Series, current_iv: float) -> Optional[float]:
    """Proxy IV percentile: rank of current IV within the trailing 1y 20-day realized-vol range."""
    close = daily_close.dropna()
    if len(close) < 60 or current_iv is None:
        return None
    logret = np.log(close / close.shift(1)).dropna()
    rv = logret.rolling(20).std() * math.sqrt(252)
    rv = rv.dropna().iloc[-252:]
    if len(rv) < 20:
        return None
    return float((rv < current_iv).mean() * 100.0)


def suggest_leaps(
    ticker: str,
    spot: float,
    rate: float,
    target_delta: float = 0.80,
    min_dte: int = 300,
) -> Optional[dict]:
    """Pick a long-dated expiry (>= ~min_dte) and the ITM call closest to target delta."""
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options
    except Exception:
        return None
    if not expiries:
        return None
    # Prefer the first expiry beyond min_dte; else the longest available.
    expiry = next((e for e in expiries if _dte(e) >= min_dte), expiries[-1])
    dte = _dte(expiry)
    try:
        calls = tk.option_chain(expiry).calls
    except Exception:
        return None
    if calls is None or calls.empty:
        return None

    t_years = max(dte, 1) / 365.0
    calls = calls.copy()
    calls["iv"] = pd.to_numeric(calls["impliedVolatility"], errors="coerce")
    calls = calls[calls["iv"] > 0]
    if calls.empty:
        return None
    calls["delta"] = calls.apply(
        lambda r: bs_call_delta(spot, float(r["strike"]), t_years, rate, float(r["iv"])), axis=1
    )
    calls = calls.dropna(subset=["delta"])
    if calls.empty:
        return None
    pick = calls.loc[(calls["delta"] - target_delta).abs().idxmin()]

    bid, ask = float(pick.get("bid", 0) or 0), float(pick.get("ask", 0) or 0)
    mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else float(pick.get("lastPrice", 0) or 0)
    spread_pct = ((ask - bid) / mid) if (mid > 0 and ask > 0 and bid > 0) else None
    return {
        "expiry": expiry,
        "dte": dte,
        "strike": float(pick["strike"]),
        "delta": float(pick["delta"]),
        "iv": float(pick["iv"]),
        "bid": bid,
        "ask": ask,
        "spread_pct": spread_pct,
        "open_interest": int(pick.get("openInterest", 0) or 0),
    }
