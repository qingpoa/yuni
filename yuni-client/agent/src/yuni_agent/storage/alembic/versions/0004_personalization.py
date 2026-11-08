"""Add companion description and the local user's profile."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from yuni_agent.storage.models import LocalProfile

revision = "0004_personalization"
down_revision = "0003_companion_gender"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("companions")}
    if "custom_description" not in columns:
        op.add_column("companions", sa.Column("custom_description", sa.Text(), nullable=False, server_default=""))
    LocalProfile.__table__.create(bind, checkfirst=True)


def downgrade() -> None:
    op.drop_table("local_profiles")
    op.drop_column("companions", "custom_description")
