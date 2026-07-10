"""Volatility Risk Premium (VRP) engine.

For each ticker: 30-day implied vol (from the options chain) vs 30-day realized vol.
- VRP        = IV30 − RV30
- IV/RV ratio = IV30 / RV30   (> 1 ⇒ options priced richer than the stock is moving)
- IV-rank proxy = where current IV sits in the trailing 1-yr realized-vol range (labeled proxy)

Ranked best→worst by the IV/RV ratio (option-seller's richness).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .options_iv import _atm_row, _dte, realized_vol_percentile
from .prices import batch_download_closes

Progress = Optional[Callable[[float, str], None]]


@dataclass
class VRPParams:
    rv_window: int = 21          # ~1 calendar month of trading days
    target_dte: int = 30
    price_period: str = "1y"     # enough for RV + 1-yr rank proxy
    max_workers: int = 8
    chunk_size: int = 300


@dataclass
class VRPCounts:
    universe: int = 0
    had_prices: int = 0
    had_iv: int = 0
    notes: list = field(default_factory=list)


def realized_vol(daily_close: pd.Series, window: int = 21) -> Optional[float]:
    """Annualized realized volatility over the last ``window`` trading days."""
    close = daily_close.dropna()
    logret = np.log(close / close.shift(1)).dropna()
    if len(logret) < window:
        return None
    vol = float(logret.iloc[-window:].std(ddof=1) * math.sqrt(252))
    return vol if (np.isfinite(vol) and vol > 0) else None


def _expiry_atm_iv(tk: yf.Ticker, expiry: str, spot: float) -> Optional[float]:
    try:
        chain = tk.option_chain(expiry)
    except Exception:
        return None
    ivs = []
    for side in (chain.calls, chain.puts):
        row = _atm_row(side, spot)
        if row is not None and pd.notna(row.get("impliedVolatility")) and row["impliedVolatility"] > 0:
            ivs.append(float(row["impliedVolatility"]))
    return float(np.mean(ivs)) if ivs else None


def iv_30d(ticker: str, spot: float, target_dte: int = 30) -> Optional[dict]:
    """ATM IV at ~30 DTE, variance-interpolated between the two bracketing expiries."""
    try:
        tk = yf.Ticker(ticker)
        expiries = tk.options
    except Exception:
        return None
    dtes = [(e, _dte(e)) for e in (expiries or [])]
    dtes = [(e, d) for e, d in dtes if d >= 0]
    if not dtes:
        return None

    below = [(e, d) for e, d in dtes if d <= target_dte]
    above = [(e, d) for e, d in dtes if d >= target_dte]

    if below and above:
        e1, d1 = below[-1]
        e2, d2 = above[0]
        if e1 == e2 or d1 == d2:
            iv = _expiry_atm_iv(tk, e1, spot)
            return {"iv": iv, "dte": d1, "method": "single"} if iv else None
        s1, s2 = _expiry_atm_iv(tk, e1, spot), _expiry_atm_iv(tk, e2, spot)
        if s1 is None and s2 is None:
            return None
        if s1 is None:
            return {"iv": s2, "dte": d2, "method": "single"}
        if s2 is None:
            return {"iv": s1, "dte": d1, "method": "single"}
        w = (target_dte - d1) / (d2 - d1)
        iv = math.sqrt(max(s1 * s1 + (s2 * s2 - s1 * s1) * w, 0.0))  # linear in variance
        return {"iv": iv, "dte": target_dte, "method": "interp"}

    e, d = above[0] if above else below[-1]
    iv = _expiry_atm_iv(tk, e, spot)
    return {"iv": iv, "dte": d, "method": "single"} if iv else None


def classify(ratio: Optional[float], iv_rank: Optional[float]) -> tuple[str, str]:
    """Return (premium_verdict, iv_regime)."""
    verdict = "Good (rich)" if (ratio is not None and ratio > 1) else "Poor (cheap)"
    if iv_rank is None:
        regime = "Unknown"
    elif iv_rank > 50:
        regime = "Rich"
    elif iv_rank < 30:
        regime = "Cheap"
    else:
        regime = "Neutral"
    return verdict, regime


def run_vrp_scan(
    universe_df: pd.DataFrame,
    params: VRPParams = VRPParams(),
    progress: Progress = None,
) -> tuple[pd.DataFrame, VRPCounts]:
    """Compute VRP for every ticker in the universe and rank by IV/RV descending."""
    counts = VRPCounts()

    def _p(frac, msg):
        if progress:
            progress(min(max(frac, 0.0), 1.0), msg)

    tickers = list(dict.fromkeys(universe_df["ticker"].tolist()))
    counts.universe = len(tickers)
    meta = universe_df.drop_duplicates("ticker").set_index("ticker")

    _p(0.03, "Downloading price history…")
    closes = batch_download_closes(tickers, period=params.price_period,
                                   chunk_size=params.chunk_size, progress=progress,
                                   progress_span=(0.05, 0.45), min_bars=params.rv_window + 5)
    counts.had_prices = len(closes)
    if not closes:
        return pd.DataFrame(), counts

    # Threaded IV30 fetch (options endpoint per ticker).
    spots = {t: float(s.iloc[-1]) for t, s in closes.items()}
    iv_map: dict[str, dict] = {}
    items = list(spots.items())
    done = 0
    with ThreadPoolExecutor(max_workers=params.max_workers) as ex:
        futs = {ex.submit(iv_30d, t, spot, params.target_dte): t for t, spot in items}
        for fut in as_completed(futs):
            t = futs[fut]
            done += 1
            _p(0.45 + 0.5 * done / max(len(items), 1), f"Fetching IV — {done}/{len(items)}")
            try:
                res = fut.result()
            except Exception:
                res = None
            if res and res.get("iv"):
                iv_map[t] = res

    counts.had_iv = len(iv_map)

    rows = []
    for t, ivinfo in iv_map.items():
        iv = ivinfo["iv"]
        rv = realized_vol(closes[t], params.rv_window)
        if rv is None:
            continue
        ratio = iv / rv if rv > 0 else None
        iv_rank = realized_vol_percentile(closes[t], iv)
        verdict, regime = classify(ratio, iv_rank)
        rows.append({
            "Ticker": t,
            "Company": meta.loc[t, "name"] if t in meta.index else "",
            "Sector": meta.loc[t, "sector"] if t in meta.index else "",
            "Price": spots[t],
            "IV30 %": round(iv * 100, 1),
            "RV30 %": round(rv * 100, 1),
            "VRP (IV-RV) %": round((iv - rv) * 100, 1),
            "IV/RV": round(ratio, 2) if ratio is not None else np.nan,
            "IV-rank proxy %": round(iv_rank, 0) if iv_rank is not None else np.nan,
            "IV regime": regime,
            "Premium": verdict,
            "IV DTE": ivinfo["dte"],
        })

    _p(1.0, "Done")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("IV/RV", ascending=False, na_position="last").reset_index(drop=True)
    return df, counts
