"""Staged LEAPS screening funnel (cheapest filters first).

1. Universe: US stocks with market cap / volume / price above thresholds (from leaps.universe).
2. Batch-download 2y daily OHLCV; keep tickers with weekly RSI below the threshold.
3. Survivors: keep those whose Ripster EMA clouds are stacked bullishly (fast>med>slow).
4. Survivors only: fetch ATM IV; keep IV below the threshold; add IV proxy percentile,
   a suggested LEAPS contract, and a composite LEAPS score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from .indicators import ripster_cloud_state, weekly_rsi
from .options_iv import atm_iv, realized_vol_percentile, suggest_leaps

Progress = Optional[Callable[[float, str], None]]

# LEAPS score component weights (sum to 1.0).
_W = {"rsi": 0.25, "ema": 0.20, "iv": 0.20, "ivrank": 0.15, "liquidity": 0.15, "earnings": 0.05}


@dataclass
class ScanParams:
    rsi_threshold: float = 25.0
    rsi_period: int = 14
    iv_threshold: float = 0.30
    min_market_cap: float = 2e9
    min_volume: float = 1e6
    min_price: float = 5.0
    target_delta: float = 0.80
    leaps_min_dte: int = 300
    atm_min_dte: int = 30
    max_tickers: Optional[int] = None
    chunk_size: int = 300


@dataclass
class ScanCounts:
    universe: int = 0
    after_price_data: int = 0
    after_rsi: int = 0
    after_ema: int = 0
    after_iv: int = 0
    notes: list = field(default_factory=list)


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _extract_close(raw: pd.DataFrame, ticker: str, single: bool) -> Optional[pd.Series]:
    try:
        if single:
            s = raw["Close"]
        else:
            s = raw[ticker]["Close"]
        return s.dropna() if s is not None else None
    except (KeyError, TypeError):
        return None


def _next_earnings_days(tk: yf.Ticker) -> Optional[int]:
    """Best-effort days until next earnings; None if unknown."""
    try:
        cal = tk.calendar
        dt = None
        if isinstance(cal, dict):
            vals = cal.get("Earnings Date")
            dt = (vals[0] if isinstance(vals, (list, tuple)) and vals else vals)
        elif isinstance(cal, pd.DataFrame) and "Earnings Date" in cal.index:
            dt = cal.loc["Earnings Date"].iloc[0]
        if dt is None:
            return None
        return (pd.Timestamp(dt).normalize() - pd.Timestamp.today().normalize()).days
    except Exception:
        return None


def _score(rsi, sep_pct, iv, iv_pct, spread_pct, oi, earnings_days, p: ScanParams) -> float:
    s_rsi = np.clip((p.rsi_threshold - rsi) / max(p.rsi_threshold, 1e-6), 0, 1)
    s_ema = np.clip(sep_pct / 0.10, 0, 1)
    s_iv = np.clip((p.iv_threshold - iv) / max(p.iv_threshold, 1e-6), 0, 1)
    s_ivrank = np.clip((100 - iv_pct) / 100, 0, 1) if iv_pct is not None else 0.5
    s_spread = np.clip(1 - (spread_pct / 0.15), 0, 1) if spread_pct is not None else 0.5
    s_oi = np.clip(oi / 1000, 0, 1)
    s_liq = (s_spread + s_oi) / 2
    if earnings_days is None:
        s_earn = 0.7
    else:
        s_earn = 0.3 if 0 <= earnings_days <= 14 else 1.0
    total = (_W["rsi"] * s_rsi + _W["ema"] * s_ema + _W["iv"] * s_iv +
             _W["ivrank"] * s_ivrank + _W["liquidity"] * s_liq + _W["earnings"] * s_earn)
    return round(float(total) * 100, 1)


def run_leaps_scan(
    universe_df: pd.DataFrame,
    rate: float,
    params: ScanParams = ScanParams(),
    progress: Progress = None,
) -> tuple[pd.DataFrame, ScanCounts]:
    """Run the full funnel. Returns (results_df, counts)."""
    counts = ScanCounts()

    def _p(frac, msg):
        if progress:
            progress(min(max(frac, 0.0), 1.0), msg)

    tickers = list(dict.fromkeys(universe_df["ticker"].tolist()))
    if params.max_tickers:
        tickers = tickers[: params.max_tickers]
    counts.universe = len(tickers)
    meta = universe_df.drop_duplicates("ticker").set_index("ticker")

    # --- Stages 2 & 3: price download -> weekly RSI -> EMA cloud ---------------------
    survivors: list[dict] = []
    chunks = list(_chunks(tickers, params.chunk_size))
    for ci, chunk in enumerate(chunks):
        _p(0.05 + 0.55 * ci / max(len(chunks), 1),
           f"Downloading prices & scanning RSI/EMA — batch {ci + 1}/{len(chunks)}")
        single = len(chunk) == 1
        try:
            raw = yf.download(chunk, period="2y", interval="1d", group_by="ticker",
                              auto_adjust=True, threads=True, progress=False)
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        for t in chunk:
            close = _extract_close(raw, t, single)
            if close is None or len(close) < 60:
                continue
            counts.after_price_data += 1
            rsi = weekly_rsi(close, params.rsi_period)
            if rsi is None or rsi >= params.rsi_threshold:
                continue
            counts.after_rsi += 1
            cloud = ripster_cloud_state(close)
            if cloud is None or not cloud.bullish_stack:
                continue
            counts.after_ema += 1
            survivors.append({"ticker": t, "close": close, "rsi": rsi,
                              "sep_pct": cloud.separation_pct, "price": float(close.iloc[-1])})

    # --- Stage 4: options / IV on survivors -----------------------------------------
    rows: list[dict] = []
    for si, s in enumerate(survivors):
        _p(0.60 + 0.38 * si / max(len(survivors), 1),
           f"Fetching IV & LEAPS contracts — {si + 1}/{len(survivors)}")
        t, spot = s["ticker"], s["price"]
        iv_info = atm_iv(t, spot, params.atm_min_dte)
        if not iv_info or iv_info["iv"] >= params.iv_threshold:
            continue
        counts.after_iv += 1
        iv = iv_info["iv"]
        iv_pct = realized_vol_percentile(s["close"], iv)
        leaps = suggest_leaps(t, spot, rate, params.target_delta, params.leaps_min_dte)
        try:
            earn_days = _next_earnings_days(yf.Ticker(t))
        except Exception:
            earn_days = None

        spread_pct = leaps.get("spread_pct") if leaps else None
        oi = leaps.get("open_interest", 0) if leaps else 0
        score = _score(s["rsi"], s["sep_pct"], iv, iv_pct, spread_pct, oi, earn_days, params)

        rows.append({
            "Ticker": t,
            "Company": meta.loc[t, "name"] if t in meta.index else "",
            "Sector": meta.loc[t, "sector"] if t in meta.index else "",
            "Price": spot,
            "Mkt Cap": float(meta.loc[t, "market_cap"]) if t in meta.index else np.nan,
            "Weekly RSI": round(s["rsi"], 1),
            "EMA sep %": round(s["sep_pct"] * 100, 2),
            "ATM IV %": round(iv * 100, 1),
            "IV pct (proxy)": round(iv_pct, 0) if iv_pct is not None else np.nan,
            "LEAPS expiry": leaps["expiry"] if leaps else "—",
            "LEAPS strike": leaps["strike"] if leaps else np.nan,
            "LEAPS delta": round(leaps["delta"], 2) if leaps else np.nan,
            "Spread %": round(spread_pct * 100, 1) if spread_pct is not None else np.nan,
            "Open int": oi,
            "Earnings in (d)": earn_days if earn_days is not None else np.nan,
            "LEAPS score": score,
        })

    _p(1.0, "Done")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("LEAPS score", ascending=False).reset_index(drop=True)
    return df, counts
