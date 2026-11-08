"""Preserve validated reply parts for history and idempotent replay."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_reply_parts"
down_revision = "0004_personalization"
branch_labels = None
depends_on = None


def upgrade() -> None:
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("chat_messages")}
    if "reply_parts" not in columns:
        op.add_column("chat_messages", sa.Column("reply_parts", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("chat_messages", "reply_parts")
