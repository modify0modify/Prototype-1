"""Safety guards for the monitoring dashboard.

Hard rules
----------
* Never activate live trading from the UI.
* Always label simulated / non-live data clearly.
* Never expose secrets (API keys, tokens, passwords, acknowledgements).
"""

from __future__ import annotations

from typing import Any, Literal

from rl_trader.config.masking import mask_account_id
from rl_trader.storage.secrets import (
    FORBIDDEN_COLUMN_SUBSTRINGS,
    SENSITIVE_CONFIG_KEYS,
    sanitize_config_for_storage,
)

DataMode = Literal["research", "backtest", "paper", "live", "unknown"]

MODE_LABELS: dict[DataMode, str] = {
    "research": "RESEARCH (offline / simulated)",
    "backtest": "BACKTEST (historical simulation)",
    "paper": "PAPER (simulated broker — not live)",
    "live": "LIVE (read-only placeholder — activation disabled)",
    "unknown": "UNKNOWN SOURCE",
}

SIMULATED_BANNER = (
    "SIMULATED RESULTS — Paper and backtest figures are not live trading. "
    "This dashboard cannot enable live trading."
)

LIVE_ACTIVATION_BLOCKED_MESSAGE = (
    "Live-trading activation is disabled in the monitoring dashboard. "
    "Use gated CLI / config activation outside this UI (not available here)."
)


def mode_label(mode: str | None) -> str:
    """Human-readable operating-mode badge."""
    key: DataMode = "unknown"
    if mode == "research":
        key = "research"
    elif mode == "backtest":
        key = "backtest"
    elif mode == "paper":
        key = "paper"
    elif mode == "live":
        key = "live"
    return MODE_LABELS[key]


def is_simulated_mode(mode: str | None) -> bool:
    """True when results should carry a simulated banner."""
    return mode in {None, "research", "backtest", "paper", "unknown"}


def redact_mapping(data: Any) -> Any:
    """Recursively redact secrets and mask account identifiers for display."""
    cleaned = sanitize_config_for_storage(data)
    return _mask_accounts(cleaned)


def _mask_accounts(data: Any) -> Any:
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            key_l = str(key).lower()
            if key_l in {"account_id", "live_account_id"}:
                out[str(key)] = mask_account_id(str(value) if value is not None else None)
            elif key_l in SENSITIVE_CONFIG_KEYS or any(
                frag in key_l for frag in FORBIDDEN_COLUMN_SUBSTRINGS
            ):
                out[str(key)] = "<redacted>"
            else:
                out[str(key)] = _mask_accounts(value)
        return out
    if isinstance(data, list):
        return [_mask_accounts(v) for v in data]
    return data


def assert_no_live_activation_controls(page_source_flags: dict[str, bool]) -> None:
    """Test helper: refuse dashboard flags that would enable live trading."""
    forbidden = {
        "enable_live_trading",
        "live_trading_enabled",
        "activate_live",
        "submit_live_order",
    }
    for name in forbidden:
        if page_source_flags.get(name):
            raise RuntimeError(f"dashboard must not expose {name}")


def contains_secret_plaintext(text: str) -> bool:
    """Heuristic: True if text looks like it embeds raw secret material."""
    lowered = text.lower()
    suspicious = (
        "sk-",
        "bearer ",
        "api_key=",
        "api-secret",
        "password=",
        "-----begin",
    )
    # Placeholder markers alone are OK; still flag if a secret pattern remains.
    return any(s in lowered for s in suspicious)
