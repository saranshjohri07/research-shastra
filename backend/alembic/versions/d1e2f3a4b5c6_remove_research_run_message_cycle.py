"""Remove circular research run/message foreign key

Revision ID: d1e2f3a4b5c6
Revises: c9d0e1f2a3b4
Create Date: 2026-11-24

"""
from typing import Sequence, Union

from alembic import op


revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, Sequence[str], None] = "c9d0e1f2a3b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(
        "research_runs_message_id_fkey",
        "research_runs",
        type_="foreignkey",
    )


def downgrade() -> None:
    op.create_foreign_key(
        "research_runs_message_id_fkey",
        "research_runs",
        "messages",
        ["message_id"],
        ["id"],
        ondelete="SET NULL",
    )
