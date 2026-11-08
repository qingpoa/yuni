"""Add local reminders."""

from __future__ import annotations

from alembic import op

from yuni_agent.storage.models import Reminder

revision = "0002_reminders"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    Reminder.__table__.create(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    Reminder.__table__.drop(bind=op.get_bind(), checkfirst=True)
