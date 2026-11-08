"""Structured sticker semantics: emotions for understanding, usage for future selection."""
import sqlalchemy as sa
from alembic import op

revision = "0014_sticker_semantics"
down_revision = "0013_sticker_messages"
branch_labels = None
depends_on = None


def upgrade():
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("stickers")}
    if "emotions" not in columns:
        op.add_column("stickers", sa.Column("emotions", sa.JSON(), nullable=False, server_default="[]"))
    if "usage" not in columns:
        op.add_column("stickers", sa.Column("usage", sa.String(300), nullable=False, server_default=""))


def downgrade():
    raise RuntimeError("Restore a matching pre-upgrade backup")
