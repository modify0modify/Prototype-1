"""Modular reinforcement-learning stock-trading platform (skeleton).

Safety contract
---------------
- RL agents emit *target portfolios* only; they never send broker orders.
- A separate deterministic risk engine approves, rejects, or resizes targets.
- Default runtime mode is paper trading.
- Live deployment requires explicit multi-check activation (see ``rl_trading.live``).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
