"""Shared configuration constants (including live acknowledgement phrase)."""

from __future__ import annotations

# Exact phrase required in LIVE_TRADING_ACKNOWLEDGEMENT before any live broker connect.
# Must match character-for-character (no extra whitespace).
LIVE_TRADING_CONFIRMATION_PHRASE = "I_UNDERSTAND_THIS_ENABLES_LIVE_BROKER_ORDERS"

# Skeleton allow-list for universe validation (extend as data coverage grows).
SUPPORTED_TICKERS: frozenset[str] = frozenset(
    {
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "JPM",
        "XOM",
        "JNJ",
        "SPY",
        "QQQ",
        "IWM",
        "DIA",
        "VTI",
    }
)
