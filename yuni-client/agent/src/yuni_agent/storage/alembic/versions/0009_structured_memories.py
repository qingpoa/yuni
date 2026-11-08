"""Structured memories, source timestamps, entities and lossless pending candidates."""
import sqlalchemy as sa
from alembic import op

revision = "0009_structured_memories"
down_revision = "0008_chat_batches"
branch_labels = None
depends_on = None


def upgrade():
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    if "memories" not in existing:
        # Historical partial installations may not have initialized memory storage.
        return
    if "memory_entities" not in existing:
        _create_entities()
    columns = {item["name"] for item in sa.inspect(op.get_bind()).get_columns("memories")}
    _upgrade_columns(columns)


def _create_entities():
    op.create_table(
        "memory_entities",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("companion_id", sa.Integer(), sa.ForeignKey("companions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("entity_type", sa.String(20), nullable=False, server_default="person"),
        sa.Column("canonical_name", sa.String(100), nullable=False),
        sa.Column("relation_to_user", sa.String(100)),
        sa.Column("aliases", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("merged_into_id", sa.Integer(), sa.ForeignKey("memory_entities.id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.current_timestamp()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.current_timestamp()),
    )
    op.create_index("ix_memory_entities_companion", "memory_entities", ["companion_id", "status"])


def _upgrade_columns(columns):
    # SQLite supports ADD COLUMN REFERENCES for nullable columns without rebuilding
    # the self-referencing memories table (which would cascade-delete its sources).
    for column in [
        sa.Column("kind", sa.String(20), nullable=False, server_default="fact"),
        sa.Column("domain", sa.String(20), nullable=False, server_default="real_world"),
        sa.Column("scene_key", sa.String(200)),
        sa.Column("qualifier", sa.String(200), nullable=False, server_default=""),
        sa.Column("event_status", sa.String(20)),
        sa.Column("outcome", sa.String(1000)),
        sa.Column("time_text", sa.String(200)),
        sa.Column("time_precision", sa.String(20), nullable=False, server_default="unknown"),
        sa.Column("fresh_until", sa.DateTime(timezone=True)),
        sa.Column("time_meta", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("entity_ids", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("suppressed_until_message_id", sa.Integer()),
    ]:
        if column.name not in columns:
            op.add_column("memories", column)
    if "subject_entity_id" not in columns:
        op.execute("ALTER TABLE memories ADD COLUMN subject_entity_id INTEGER REFERENCES memory_entities(id) ON DELETE SET NULL")
    if "conflict_with_id" not in columns:
        op.execute("ALTER TABLE memories ADD COLUMN conflict_with_id INTEGER REFERENCES memories(id) ON DELETE SET NULL")
    source_columns = {item["name"] for item in sa.inspect(op.get_bind()).get_columns("memory_sources")}
    for name in ("message_sent_at", "created_at"):
        if name not in source_columns:
            op.add_column("memory_sources", sa.Column(name, sa.DateTime(timezone=True)))
    candidate_columns = {item["name"] for item in sa.inspect(op.get_bind()).get_columns("memory_candidates")}
    if "structured_data" not in candidate_columns:
        op.add_column("memory_candidates", sa.Column("structured_data", sa.JSON(), nullable=False, server_default="{}"))
    if "kind" not in columns:
        op.execute("UPDATE memories SET kind='event' WHERE type='important_event'")
    message_columns = {item["name"] for item in sa.inspect(op.get_bind()).get_columns("chat_messages")}
    if "created_at" in message_columns:
        op.execute("UPDATE memory_sources SET message_sent_at=(SELECT created_at FROM chat_messages WHERE id=message_id), created_at=CURRENT_TIMESTAMP")
    op.execute("UPDATE memories SET suppressed_until_message_id=(SELECT MAX(cm.id) FROM chat_messages cm JOIN conversations c ON c.id=cm.conversation_id WHERE c.companion_id=memories.companion_id) WHERE status='deleted' OR deleted_at IS NOT NULL")
    indexes = {item["name"] for item in sa.inspect(op.get_bind()).get_indexes("memories")}
    if "ix_memories_scope" not in indexes:
        op.create_index("ix_memories_scope", "memories", ["companion_id", "status", "kind", "domain"])
    if "ix_memories_slot" not in indexes:
        op.create_index("ix_memories_slot", "memories", ["companion_id", "domain", "scene_key", "subject_entity_id", "subject_key", "qualifier", "status"])


def downgrade():
    raise RuntimeError("Restore a matching pre-upgrade backup to avoid losing structured memory data")
