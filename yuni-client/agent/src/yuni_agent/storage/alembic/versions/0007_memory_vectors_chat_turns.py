"""Durable turn completion and rebuildable factual memory embeddings."""
import sqlalchemy as sa
from alembic import op

revision = "0007_memory_vectors_chat_turns"
down_revision = "0006_expressions"
branch_labels = None
depends_on = None


def upgrade():
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    if "memory_embeddings" not in existing:
        op.create_table("memory_embeddings",
            sa.Column("memory_id", sa.Integer(), sa.ForeignKey("memories.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("profile", sa.String(64), nullable=False),
            sa.Column("model", sa.String(120), nullable=False),
            sa.Column("dimension", sa.Integer(), nullable=False),
            sa.Column("content_hash", sa.String(64), nullable=False),
            sa.Column("vector", sa.LargeBinary(), nullable=False))
    if "chat_turns" not in existing:
        op.create_table("chat_turns",
            sa.Column("user_message_id", sa.Integer(), sa.ForeignKey("chat_messages.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("status", sa.String(20), nullable=False))


def downgrade():
    op.drop_table("chat_turns")
    op.drop_table("memory_embeddings")
