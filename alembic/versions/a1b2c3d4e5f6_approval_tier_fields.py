"""Add formal approval tier fields to model_approvals.

Revision ID: a1b2c3d4e5f6
Revises: 76c66464d171
Create Date: 2026-10-17 10:55:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "76c66464d171"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("model_approvals", schema=None) as batch_op:
        batch_op.alter_column(
            "checkpoint_id",
            existing_type=sa.Uuid(),
            nullable=True,
        )
        batch_op.add_column(
            sa.Column("approval_tier", sa.String(length=16), nullable=False, server_default="paper")
        )
        batch_op.add_column(
            sa.Column("content_hash", sa.String(length=128), nullable=False, server_default="")
        )
        batch_op.add_column(
            sa.Column("code_commit_id", sa.String(length=64), nullable=False, server_default="")
        )
        batch_op.add_column(
            sa.Column("evidence_json", sa.Text(), nullable=False, server_default="{}")
        )


def downgrade() -> None:
    with op.batch_alter_table("model_approvals", schema=None) as batch_op:
        batch_op.drop_column("evidence_json")
        batch_op.drop_column("code_commit_id")
        batch_op.drop_column("content_hash")
        batch_op.drop_column("approval_tier")
        batch_op.alter_column(
            "checkpoint_id",
            existing_type=sa.Uuid(),
            nullable=False,
        )
