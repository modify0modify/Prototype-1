"""Enforce that the storage schema never stores secrets.

Forbidden column name substrings (case-insensitive): api_key, api_secret,
password, token, private_key, access_key, secret_key, bearer, credential.
Configuration JSON must be sanitized via :func:`sanitize_config_for_storage`
before insert into ``configuration_versions``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import inspect

from rl_trader.storage.db import Base

FORBIDDEN_COLUMN_SUBSTRINGS: tuple[str, ...] = (
    "api_key",
    "api_secret",
    "password",
    "passwd",
    "token",
    "private_key",
    "access_key",
    "secret_key",
    "bearer",
    "credential",
    "auth_header",
)

# Nested config keys stripped before persisting configuration_versions.
SENSITIVE_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "api_secret",
        "paper_api_key",
        "paper_api_secret",
        "live_api_key",
        "live_api_secret",
        "password",
        "token",
        "secret",
        "acknowledgement",  # live confirmation phrase — treat as sensitive
    }
)


class SecretColumnError(RuntimeError):
    """Raised when a mapped column name looks like a secret field."""


def assert_no_secret_columns(base: type[Base] = Base) -> None:
    """Fail if any ORM column name contains a forbidden secret substring."""
    # Ensure models are registered.
    import rl_trader.storage.models  # noqa: F401

    offenders: list[str] = []
    for table in base.metadata.tables.values():
        for col in table.columns:
            name = col.name.lower()
            for frag in FORBIDDEN_COLUMN_SUBSTRINGS:
                if frag in name:
                    offenders.append(f"{table.name}.{col.name}")
                    break
    if offenders:
        raise SecretColumnError(
            "secret-like columns forbidden in storage schema: " + ", ".join(sorted(offenders))
        )


def sanitize_config_for_storage(data: Any) -> Any:
    """Recursively drop sensitive keys from a config mapping before DB insert."""
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            key_l = str(key).lower()
            if key_l in SENSITIVE_CONFIG_KEYS or any(
                frag in key_l for frag in FORBIDDEN_COLUMN_SUBSTRINGS
            ):
                continue
            out[str(key)] = sanitize_config_for_storage(value)
        return out
    if isinstance(data, list):
        return [sanitize_config_for_storage(v) for v in data]
    return data


def mapped_column_names() -> set[str]:
    """Return ``table.column`` names for integrity tests."""
    import rl_trader.storage.models  # noqa: F401

    names: set[str] = set()
    for table in Base.metadata.tables.values():
        for col in inspect(table).c:
            names.add(f"{table.name}.{col.name}")
    return names
