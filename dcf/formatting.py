"""Display formatting helpers for currency, large numbers, and percentages."""

from __future__ import annotations

from typing import Optional


def fmt_currency(value: Optional[float], currency: str = "USD", decimals: int = 2) -> str:
    if value is None:
        return "—"
    sign = "-" if value < 0 else ""
    return f"{sign}{_symbol(currency)}{abs(value):,.{decimals}f}"


def fmt_big(value: Optional[float], currency: str = "USD") -> str:
    """Compact large-number format: $1.23T / $45.6B / $789.0M."""
    if value is None:
        return "—"
    sym = _symbol(currency)
    sign = "-" if value < 0 else ""
    v = abs(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= threshold:
            return f"{sign}{sym}{v / threshold:,.2f}{suffix}"
    return f"{sign}{sym}{v:,.0f}"


def fmt_pct(value: Optional[float], decimals: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{decimals}f}%"


def fmt_multiple(value: Optional[float]) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}x"


def _symbol(currency: str) -> str:
    return {
        "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥",
        "CNY": "¥", "HKD": "HK$", "CAD": "C$", "AUD": "A$",
    }.get((currency or "USD").upper(), f"{currency} ")
