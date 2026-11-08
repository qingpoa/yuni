"""Initial local storage schema."""

from __future__ import annotations

from alembic import op

from yuni_agent.storage.models import Base

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)
    bind.exec_driver_sql(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts "
        "USING fts5(title, content, content='memories', content_rowid='id')"
    )
    bind.exec_driver_sql(
        "CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN "
        "INSERT INTO memories_fts(rowid, title, content) VALUES (new.id, new.title, new.content); END"
    )
    bind.exec_driver_sql(
        "CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN "
        "INSERT INTO memories_fts(memories_fts, rowid, title, content) "
        "VALUES ('delete', old.id, old.title, old.content); END"
    )
    bind.exec_driver_sql(
        "CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN "
        "INSERT INTO memories_fts(memories_fts, rowid, title, content) "
        "VALUES ('delete', old.id, old.title, old.content); "
        "INSERT INTO memories_fts(rowid, title, content) VALUES (new.id, new.title, new.content); END"
    )


def downgrade() -> None:
    bind = op.get_bind()
    bind.exec_driver_sql("DROP TRIGGER IF EXISTS memories_au")
    bind.exec_driver_sql("DROP TRIGGER IF EXISTS memories_ad")
    bind.exec_driver_sql("DROP TRIGGER IF EXISTS memories_ai")
    bind.exec_driver_sql("DROP TABLE IF EXISTS memories_fts")
    Base.metadata.drop_all(bind=bind)
