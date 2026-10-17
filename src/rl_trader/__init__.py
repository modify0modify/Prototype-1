"""Modular reinforcement-learning stock-trading system (architecture skeleton).

Safety contract
---------------
- RL agents emit *target portfolios* only; they never send broker orders.
- A separate deterministic risk engine approves, rejects, or resizes targets.
- Default runtime mode is paper trading.
- Live deployment requires explicit multi-check activation.

Import boundaries
-----------------
Research modules must not import live execution or live broker adapters.
See ``rl_trader.utils.boundaries`` and the README dependency diagram.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
