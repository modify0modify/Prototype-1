"""Execution orchestration.

Paper execution is the default safe path. Live execution lives in
``rl_trader.execution.live`` and must not be imported by research modules.
"""

from rl_trader.execution.paper import PaperExecutor, PaperTradingDisabledError

__all__ = ["PaperExecutor", "PaperTradingDisabledError"]
