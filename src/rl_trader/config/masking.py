"""Secret-safe formatting helpers — never log raw secrets or full account IDs."""

from __future__ import annotations

from pydantic import SecretStr


def mask_account_id(account_id: str | None) -> str:
    """Mask an account identifier for logs / repr (keep last 4 chars when long enough)."""
    if not account_id:
        return "<empty>"
    value = account_id.strip()
    if len(value) <= 4:
        return "****"
    return f"***{value[-4:]}"


def secret_is_set(secret: SecretStr | None) -> bool:
    """True when a secret is present and non-empty (without logging the value)."""
    if secret is None:
        return False
    return bool(secret.get_secret_value().strip())


def redact_secret(secret: SecretStr | None) -> str:
    """Return a non-sensitive placeholder for logging."""
    return "<set>" if secret_is_set(secret) else "<unset>"
