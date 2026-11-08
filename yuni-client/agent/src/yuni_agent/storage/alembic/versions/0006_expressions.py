"""User-sourced expression learning and rebuildable vector projections."""
import sqlalchemy as sa
from alembic import op

revision = "0006_expressions"
down_revision = "0005_reply_parts"
branch_labels = None
depends_on = None


def upgrade():
    # The original bootstrap migration creates current metadata for fresh installs.
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    if {"expressions", "expression_sources", "expression_progress", "expression_vectors"} <= existing:
        return
    op.create_table("expressions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("companion_id", sa.Integer(), sa.ForeignKey("companions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("situation", sa.String(100), nullable=False),
        sa.Column("style", sa.String(100), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("companion_id", "situation", "style", name="uq_expression_scope"))
    op.create_index("ix_expressions_companion_id", "expressions", ["companion_id"])
    op.create_table("expression_sources",
        sa.Column("expression_id", sa.Integer(), sa.ForeignKey("expressions.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("message_id", sa.Integer(), sa.ForeignKey("chat_messages.id", ondelete="CASCADE"), primary_key=True))
    op.create_table("expression_progress",
        sa.Column("conversation_id", sa.Integer(), sa.ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("until_message_id", sa.Integer(), nullable=False),
        sa.Column("last_skipped_message_id", sa.Integer(), nullable=True))
    op.create_table("expression_vectors",
        sa.Column("expression_id", sa.Integer(), sa.ForeignKey("expressions.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("profile", sa.String(64), nullable=False),
        sa.Column("model", sa.String(120), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("vector", sa.JSON(), nullable=False))


def downgrade():
    for table in ("expression_vectors", "expression_progress", "expression_sources", "expressions"):
        op.drop_table(table)
