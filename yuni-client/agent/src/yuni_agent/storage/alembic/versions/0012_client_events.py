"""Persist application events for global SSE replay."""
import sqlalchemy as sa
from alembic import op

revision = "0012_client_events"
down_revision = "0011_proactive_rounds"
branch_labels = None
depends_on = None


def upgrade():
    if "client_events" not in sa.inspect(op.get_bind()).get_table_names():
        op.create_table(
            "client_events",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("event_type", sa.String(50), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("dedupe_key", sa.String(200), unique=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sqlite_autoincrement=True,
        )


def downgrade():
    raise RuntimeError("Restore a matching pre-upgrade backup")
