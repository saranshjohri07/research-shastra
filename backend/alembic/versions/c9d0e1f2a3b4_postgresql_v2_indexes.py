"""Add PostgreSQL V2 relationship indexes

Revision ID: c9d0e1f2a3b4
Revises: b7c8d9e0f1a2
Create Date: 2026-11-24

"""
from typing import Sequence, Union

from alembic import op


revision: str = "c9d0e1f2a3b4"
down_revision: Union[str, Sequence[str], None] = "b7c8d9e0f1a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_research_runs_conversation_id",
        "research_runs",
        ["conversation_id"],
    )
    op.create_index(
        "ix_research_runs_created_at",
        "research_runs",
        ["created_at"],
    )
    op.create_index(
        "ix_research_runs_message_id",
        "research_runs",
        ["message_id"],
    )
    op.create_index(
        "ix_research_runs_thread_id",
        "research_runs",
        ["thread_id"],
    )
    op.create_index(
        "ix_trace_events_research_run_id",
        "trace_events",
        ["research_run_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_trace_events_research_run_id", table_name="trace_events")
    op.drop_index("ix_research_runs_thread_id", table_name="research_runs")
    op.drop_index("ix_research_runs_message_id", table_name="research_runs")
    op.drop_index("ix_research_runs_created_at", table_name="research_runs")
    op.drop_index("ix_research_runs_conversation_id", table_name="research_runs")
