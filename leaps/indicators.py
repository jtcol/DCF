"""Technical indicators for the LEAPS screen: weekly RSI and Ripster EMA clouds.

RSI uses Wilder's smoothing via ``ewm(com=period-1)`` (matches the standard RSI).
Ripster clouds are pairs of EMAs; the bullish stack requires the fast cloud to sit
entirely above the med cloud, which sits entirely above the slow cloud (no overlap).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Ripster "classic" cloud EMA pairs.
FAST_CLOUD = (5, 12)
MED_CLOUD = (34, 50)
SLOW_CLOUD = (72, 89)
_ALL_EMAS = sorted({*FAST_CLOUD, *MED_CLOUD, *SLOW_CLOUD})


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def weekly_rsi(daily_close: pd.Series, period: int = 14) -> Optional[float]:
    """Latest weekly RSI from a daily close series (resampled to Friday weekly bars)."""
    close = daily_close.dropna()
    if len(close) < (period + 1) * 5:  # need enough daily bars for ~period weeks
        # Still attempt if we at least have a handful of weeks.
        if len(close) < period + 5:
            return None
    weekly = close.resample("W-FRI").last().dropna()
    if len(weekly) < period + 1:
        return None
    rsi = compute_rsi(weekly, period)
    val = rsi.dropna()
    return float(val.iloc[-1]) if len(val) else None


@dataclass
class CloudState:
    bullish_stack: bool
    separation_pct: float  # min gap between adjacent clouds as % of price (0 if overlapping)
    emas: dict[int, float]


def ripster_cloud_state(daily_close: pd.Series) -> Optional[CloudState]:
    """Evaluate the Ripster classic-cloud stack on the latest daily bar."""
    close = daily_close.dropna()
    if len(close) < max(_ALL_EMAS) + 5:
        return None
    emas = {p: float(close.ewm(span=p, adjust=False).mean().iloc[-1]) for p in _ALL_EMAS}
    price = float(close.iloc[-1])
    if price <= 0:
        return None

    fast_lo, fast_hi = min(emas[FAST_CLOUD[0]], emas[FAST_CLOUD[1]]), max(emas[FAST_CLOUD[0]], emas[FAST_CLOUD[1]])
    med_lo, med_hi = min(emas[MED_CLOUD[0]], emas[MED_CLOUD[1]]), max(emas[MED_CLOUD[0]], emas[MED_CLOUD[1]])
    slow_lo, slow_hi = min(emas[SLOW_CLOUD[0]], emas[SLOW_CLOUD[1]]), max(emas[SLOW_CLOUD[0]], emas[SLOW_CLOUD[1]])

    # Bullish stack: fast cloud above med cloud above slow cloud, with no overlap.
    gap_fast_med = fast_lo - med_hi
    gap_med_slow = med_lo - slow_hi
    bullish = gap_fast_med > 0 and gap_med_slow > 0
    separation = (min(gap_fast_med, gap_med_slow) / price) if bullish else 0.0
    return CloudState(bullish_stack=bullish, separation_pct=separation, emas=emas)
