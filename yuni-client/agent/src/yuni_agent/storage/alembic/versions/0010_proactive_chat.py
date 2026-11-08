"""Add proactive chat settings and conversation scheduling watermarks."""
import sqlalchemy as sa
from alembic import op

revision = "0010_proactive_chat"
down_revision = "0009_structured_memories"
branch_labels = None
depends_on = None


def upgrade():
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    if "settings" in existing:
        settings_cols = {item["name"] for item in sa.inspect(op.get_bind()).get_columns("settings")}
        for col in [
            sa.Column("proactive_chat_enabled", sa.Boolean(), nullable=False, server_default=sa.text("1")),
            sa.Column("proactive_interval_minutes", sa.Integer(), nullable=False, server_default="120"),
            sa.Column("quiet_hours_start", sa.String(5), nullable=False, server_default="23:30"),
            sa.Column("quiet_hours_end", sa.String(5), nullable=False, server_default="08:30"),
        ]:
            if col.name not in settings_cols:
                op.add_column("settings", col)
    if "conversations" in existing:
        conv_cols = {item["name"] for item in sa.inspect(op.get_bind()).get_columns("conversations")}
        for col in [
            sa.Column("next_proactive_at", sa.DateTime(timezone=True)),
            sa.Column("last_proactive_memory_id", sa.Integer()),
        ]:
            if col.name not in conv_cols:
                op.add_column("conversations", col)


def downgrade():
    raise RuntimeError("Restore a matching pre-upgrade backup")
