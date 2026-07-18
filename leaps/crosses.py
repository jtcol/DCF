"""Weekly 50/200-EMA cross detection for investment entry/exit signals.

- Death cross (exit signal, portfolio tab): 50-week EMA below the 200-week EMA, with a 3%
  buffer band — Green (no cross) / Amber (crossed, gap < 3%) / Red (gap >= 3% => exit).
- Golden cross (entry signal, scanner tab): 50-week EMA above the 200-week EMA —
  "fresh" while the gap is inside the 3% buffer, "recent" when the bullish cross happened
  within the last ~8 weeks but the gap has already run past the buffer.

All functions are pure (no Streamlit) and operate on a weekly close Series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# A 200-week EMA needs a long runway to be meaningful; below this we return None
# (insufficient history) rather than a fake signal.
MIN_WEEKLY_BARS = 210


@dataclass
class CrossState:
    ema50: float
    ema200: float
    gap: float                    # (ema50 - ema200) / ema200, signed
    weeks_since_cross: Optional[int]  # bars since the last 50/200 sign flip; None if never
    last_close: float


def weekly_ema_cross(weekly_close: pd.Series) -> Optional[CrossState]:
    """Compute the current weekly 50/200-EMA state, or None if history is too short."""
    close = weekly_close.dropna()
    if len(close) < MIN_WEEKLY_BARS:
        return None
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    e50, e200 = float(ema50.iloc[-1]), float(ema200.iloc[-1])
    if e200 <= 0:
        return None
    diff = (ema50 - ema200).to_numpy()
    sign = np.sign(diff)
    # Bars since the most recent sign change (ignoring exact-zero ties).
    weeks_since = None
    flips = np.where((sign[:-1] != 0) & (sign[1:] != 0) & (sign[:-1] != sign[1:]))[0]
    if len(flips):
        weeks_since = int(len(sign) - 1 - (flips[-1] + 1))
    return CrossState(
        ema50=e50,
        ema200=e200,
        gap=(e50 - e200) / e200,
        weeks_since_cross=weeks_since,
        last_close=float(close.iloc[-1]),
    )


def death_cross_status(state: Optional[CrossState], buffer: float = 0.03) -> str:
    """Portfolio exit flag: 'Green' | 'Amber' | 'Red' | '—' (insufficient history)."""
    if state is None:
        return "—"
    if state.gap >= 0:
        return "Green"
    if state.gap > -buffer:
        return "Amber"
    return "Red"


def golden_cross_class(
    state: Optional[CrossState],
    buffer: float = 0.03,
    recent_weeks: int = 8,
) -> Optional[str]:
    """Entry classification: 'fresh' | 'recent' | None (no qualifying golden cross)."""
    if state is None or state.gap < 0:
        return None
    if state.gap < buffer:
        return "fresh"
    if state.weeks_since_cross is not None and state.weeks_since_cross <= recent_weeks:
        return "recent"
    return None
