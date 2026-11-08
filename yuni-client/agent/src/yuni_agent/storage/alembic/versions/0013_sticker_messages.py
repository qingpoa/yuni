"""An initially empty sticker catalog and optional chat attachments."""
import sqlalchemy as sa
from alembic import op

revision = "0013_sticker_messages"
down_revision = "0012_client_events"
branch_labels = None
depends_on = None


def upgrade():
    inspector = sa.inspect(op.get_bind())
    if "stickers" not in inspector.get_table_names():
        op.create_table(
            "stickers",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("description", sa.String(500), nullable=False),
            sa.Column("relative_path", sa.String(500), nullable=False),
            sa.Column("content_type", sa.String(50), nullable=False),
            sa.Column("size_bytes", sa.Integer(), nullable=False),
            sa.Column("sha256", sa.String(64), nullable=False),
            sa.Column("enabled", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
    if "sticker" not in {column["name"] for column in inspector.get_columns("chat_messages")}:
        op.add_column("chat_messages", sa.Column("sticker", sa.JSON(), nullable=True))


def downgrade():
    raise RuntimeError("Restore a matching pre-upgrade backup")
