"""Persist the input-to-batch relationship for replay and tool provenance."""
import sqlalchemy as sa
from alembic import op

revision = "0008_chat_batches"
down_revision = "0007_memory_vectors_chat_turns"
branch_labels = None
depends_on = None


def upgrade():
    columns = {item["name"] for item in sa.inspect(op.get_bind()).get_columns("chat_turns")}
    if "batch_user_message_id" not in columns:
        with op.batch_alter_table("chat_turns") as batch:
            batch.add_column(sa.Column("batch_user_message_id", sa.Integer(), nullable=True))
            batch.create_foreign_key("fk_chat_turn_batch", "chat_messages", ["batch_user_message_id"], ["id"], ondelete="CASCADE")


def downgrade():
    with op.batch_alter_table("chat_turns") as batch:
        batch.drop_column("batch_user_message_id")
