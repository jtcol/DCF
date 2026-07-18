"""Shared batched price-history download for the LEAPS and VRP scans.

Wraps ``yf.download(group_by="ticker", threads=True)`` in chunks and returns a clean
``{ticker: close_series}`` map, so both scanners fetch prices the same way.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd
import yfinance as yf

Progress = Optional[Callable[[float, str], None]]


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _extract_close(raw: pd.DataFrame, ticker: str, single: bool = False) -> Optional[pd.Series]:
    """Pull a ticker's Close series from a yf.download frame, robust to column layout.

    yfinance may return: ticker-keyed MultiIndex (group_by='ticker', even for one ticker),
    field-keyed MultiIndex (group_by='column'), or flat columns. Handle all three.
    """
    try:
        cols = raw.columns
        if isinstance(cols, pd.MultiIndex):
            level0 = set(cols.get_level_values(0))
            if ticker in level0:                       # ('NBIS', 'Close')
                s = raw[ticker]["Close"]
            elif "Close" in level0:                    # ('Close', 'NBIS')
                sub = raw["Close"]
                s = (sub[ticker] if isinstance(sub, pd.DataFrame) and ticker in sub.columns
                     else sub.iloc[:, 0] if isinstance(sub, pd.DataFrame) else sub)
            else:
                return None
        else:
            s = raw["Close"]                           # flat columns
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s.dropna() if s is not None else None
    except (KeyError, TypeError, IndexError):
        return None


def batch_download_closes(
    tickers: list[str],
    period: str = "1y",
    chunk_size: int = 300,
    progress: Progress = None,
    progress_span: tuple[float, float] = (0.0, 1.0),
    min_bars: int = 30,
    interval: str = "1d",
) -> dict[str, pd.Series]:
    """Return {ticker: close Series} at ``interval`` for tickers with >= ``min_bars`` bars."""
    tickers = list(dict.fromkeys(tickers))
    out: dict[str, pd.Series] = {}
    chunks = list(_chunks(tickers, chunk_size))
    lo, hi = progress_span
    for ci, chunk in enumerate(chunks):
        if progress:
            frac = lo + (hi - lo) * ci / max(len(chunks), 1)
            progress(frac, f"Downloading prices — batch {ci + 1}/{len(chunks)}")
        single = len(chunk) == 1
        try:
            raw = yf.download(chunk, period=period, interval=interval, group_by="ticker",
                              auto_adjust=True, threads=True, progress=False)
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        for t in chunk:
            close = _extract_close(raw, t, single)
            if close is not None and len(close) >= min_bars:
                out[t] = close
    return out
