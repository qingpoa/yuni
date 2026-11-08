"""Optional generated image attached to an assistant chat message."""
import sqlalchemy as sa
from alembic import op

revision = "0015_generated_images"
down_revision = "0014_sticker_semantics"
branch_labels = None
depends_on = None


def upgrade():
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("chat_messages")}
    if "image" not in columns:
        op.add_column("chat_messages", sa.Column("image", sa.JSON(), nullable=True))


def downgrade():
    raise RuntimeError("Restore a matching pre-upgrade backup")
