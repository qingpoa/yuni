"""Identify every reply belonging to one proactive Planner invocation."""
import sqlalchemy as sa
from alembic import op

revision = "0011_proactive_rounds"
down_revision = "0010_proactive_chat"
branch_labels = None
depends_on = None


def upgrade():
    inspector = sa.inspect(op.get_bind())
    if "chat_messages" in inspector.get_table_names():
        columns = {column["name"] for column in inspector.get_columns("chat_messages")}
        if "proactive_round_id" not in columns:
            op.add_column("chat_messages", sa.Column("proactive_round_id", sa.String(36)))


def downgrade():
    raise RuntimeError("Restore a matching pre-upgrade backup")
