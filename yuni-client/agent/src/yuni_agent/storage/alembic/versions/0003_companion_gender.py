"""Add an optional gender to existing companion profiles."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_companion_gender"
down_revision = "0002_reminders"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The initial migration uses live metadata, so fresh databases already have the column.
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("companions")}
    if "gender" not in columns:
        op.add_column("companions", sa.Column("gender", sa.String(6), nullable=True))


def downgrade() -> None:
    op.drop_column("companions", "gender")
