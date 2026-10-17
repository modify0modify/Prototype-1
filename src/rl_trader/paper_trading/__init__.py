"""Prospective paper-trading service (approved model → risk → PaperBroker).

Never imports the live broker. Scheduled retraining is a separate hook and
does not run inside the decision loop.
"""

from rl_trader.paper_trading.market_data import StaleMarketDataError
from rl_trader.paper_trading.service import PaperTradingCycleResult, PaperTradingService

__all__ = [
    "PaperTradingCycleResult",
    "PaperTradingService",
    "StaleMarketDataError",
]
