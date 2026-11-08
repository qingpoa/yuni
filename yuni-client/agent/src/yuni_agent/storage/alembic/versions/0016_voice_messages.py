"""Optional voice message attached to an assistant chat message."""
import sqlalchemy as sa
from alembic import op

revision = "0016_voice_messages"
down_revision = "0015_generated_images"
branch_labels = None
depends_on = None


def upgrade():
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("chat_messages")}
    if "voice" not in columns:
        op.add_column("chat_messages", sa.Column("voice", sa.JSON(), nullable=True))


def downgrade():
    raise RuntimeError("Restore a matching pre-upgrade backup")
