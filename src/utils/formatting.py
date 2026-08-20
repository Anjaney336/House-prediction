from __future__ import annotations


CURRENCY_SYMBOLS = {"USD": "$", "INR": "₹", "GBP": "£", "EUR": "€", "AED": "AED ", "CAD": "C$", "AUD": "A$", "OTHER": ""}


def format_value(value: float, currency: str = "USD", compact: bool = True) -> str:
    symbol = CURRENCY_SYMBOLS.get(currency, f"{currency} ")
    absolute = abs(value)
    if compact and currency == "INR":
        if absolute >= 10_000_000:
            return f"{symbol}{value / 10_000_000:,.2f} Cr"
        if absolute >= 100_000:
            return f"{symbol}{value / 100_000:,.2f} L"
    if compact and absolute >= 1_000_000:
        return f"{symbol}{value / 1_000_000:,.2f}M"
    return f"{symbol}{value:,.2f}"
