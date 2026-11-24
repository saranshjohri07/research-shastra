"""PostgreSQL V2 application schema

Revision ID: b7c8d9e0f1a2
Revises: a1b2c3d4e5f6
Create Date: 2026-11-24

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "b7c8d9e0f1a2"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    constraints = (
        ("chunks", "chunks_paper_id_fkey"),
        ("papers", "papers_collection_id_fkey"),
        ("trace_events", "trace_events_research_run_id_fkey"),
        ("citations", "citations_research_run_id_fkey"),
        ("citations", "citations_chunk_id_fkey"),
        ("feedback", "feedback_research_run_id_fkey"),
        ("notes", "notes_paper_id_fkey"),
        ("conversations", "conversations_paper_id_fkey"),
        ("messages", "messages_conversation_id_fkey"),
    )
    for table, constraint in constraints:
        op.execute(
            sa.text(
                "ALTER TABLE IF EXISTS "
                + table
                + " DROP CONSTRAINT IF EXISTS "
                + constraint
            )
        )

    uuid_conversions = (
        ("collections", "id"),
        ("papers", "id"),
        ("papers", "collection_id"),
        ("chunks", "id"),
        ("chunks", "paper_id"),
        ("research_runs", "id"),
        ("trace_events", "id"),
        ("trace_events", "research_run_id"),
        ("citations", "id"),
        ("citations", "research_run_id"),
        ("citations", "chunk_id"),
        ("feedback", "id"),
        ("feedback", "research_run_id"),
        ("notes", "id"),
        ("notes", "paper_id"),
        ("conversations", "id"),
        ("conversations", "paper_id"),
        ("messages", "id"),
        ("messages", "conversation_id"),
    )
    for table, column in uuid_conversions:
        op.execute(
            sa.text(
                f'ALTER TABLE "{table}" ALTER COLUMN "{column}" '
                f"TYPE UUID USING NULLIF(\"{column}\"::text, '')::uuid"
            )
        )

    timestamp_columns = (
        ("collections", "created_at"),
        ("papers", "created_at"),
        ("chunks", "created_at"),
        ("research_runs", "created_at"),
        ("research_runs", "completed_at"),
        ("trace_events", "created_at"),
        ("citations", "created_at"),
        ("feedback", "created_at"),
        ("notes", "created_at"),
        ("conversations", "created_at"),
        ("conversations", "updated_at"),
        ("messages", "created_at"),
    )
    for table, column in timestamp_columns:
        op.execute(
            sa.text(
                f'ALTER TABLE "{table}" ALTER COLUMN "{column}" '
                f'TYPE TIMESTAMPTZ USING "{column}"::timestamptz'
            )
        )

    op.execute(
        "ALTER TABLE messages ALTER COLUMN response_snapshot TYPE JSONB "
        "USING CASE WHEN response_snapshot IS NULL THEN NULL "
        "WHEN btrim(response_snapshot) = '' THEN NULL "
        "ELSE response_snapshot::jsonb END"
    )
    op.execute(
        "ALTER TABLE trace_events ALTER COLUMN payload TYPE JSONB "
        "USING CASE WHEN payload IS NULL THEN NULL "
        "WHEN btrim(payload) = '' THEN NULL "
        "ELSE payload::jsonb END"
    )

    op.add_column(
        "papers",
        sa.Column("storage_key", sa.Text(), nullable=True),
    )
    op.add_column(
        "papers",
        sa.Column("mime_type", sa.String(length=100), nullable=False, server_default="application/pdf"),
    )
    op.add_column(
        "papers",
        sa.Column("index_status", sa.String(length=20), nullable=False, server_default="pending"),
    )
    op.add_column("papers", sa.Column("index_error", sa.Text(), nullable=True))
    op.add_column("papers", sa.Column("embedding_model", sa.String(length=255), nullable=True))
    op.add_column("papers", sa.Column("embedding_dimension", sa.Integer(), nullable=True))
    op.add_column(
        "papers",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    op.add_column("chunks", sa.Column("text_hash", sa.String(length=64), nullable=True))
    op.add_column("chunks", sa.Column("embedding_model", sa.String(length=255), nullable=True))
    op.add_column("chunks", sa.Column("embedding_dimension", sa.Integer(), nullable=True))
    op.add_column(
        "chunks",
        sa.Column("vector_status", sa.String(length=20), nullable=False, server_default="pending"),
    )
    op.add_column("chunks", sa.Column("vector_id", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "chunks",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    op.add_column(
        "research_runs",
        sa.Column("conversation_id", postgresql.UUID(as_uuid=False), nullable=True),
    )
    op.add_column(
        "research_runs",
        sa.Column("message_id", postgresql.UUID(as_uuid=False), nullable=True),
    )
    op.add_column("research_runs", sa.Column("answerable", sa.Boolean(), nullable=True))
    op.add_column("research_runs", sa.Column("error", sa.Text(), nullable=True))
    op.add_column("research_runs", sa.Column("abstention_reason", sa.Text(), nullable=True))
    op.add_column("research_runs", sa.Column("model_name", sa.String(length=255), nullable=True))
    op.add_column("research_runs", sa.Column("embedding_model", sa.String(length=255), nullable=True))
    op.add_column("research_runs", sa.Column("top_k", sa.Integer(), nullable=True))
    op.add_column("research_runs", sa.Column("thread_id", sa.Text(), nullable=True))
    op.add_column("research_runs", sa.Column("checkpoint_id", sa.Text(), nullable=True))
    op.add_column("research_runs", sa.Column("resume_token", sa.Text(), nullable=True))
    op.add_column(
        "research_runs",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.add_column("messages", sa.Column("research_run_id", postgresql.UUID(as_uuid=False), nullable=True))

    op.add_column("trace_events", sa.Column("thread_id", sa.Text(), nullable=True))
    op.add_column("trace_events", sa.Column("sequence", sa.Integer(), nullable=True))
    op.create_unique_constraint(
        "uq_trace_events_run_sequence", "trace_events", ["research_run_id", "sequence"]
    )

    op.add_column("citations", sa.Column("evidence_id", postgresql.UUID(as_uuid=False), nullable=True))
    op.add_column("citations", sa.Column("citation_text", sa.Text(), nullable=True))
    op.add_column("citations", sa.Column("metadata_snapshot", postgresql.JSONB(), nullable=True))
    op.create_unique_constraint(
        "uq_citations_run_index", "citations", ["research_run_id", "citation_index"]
    )

    op.add_column("notes", sa.Column("conversation_id", postgresql.UUID(as_uuid=False), nullable=True))
    op.add_column("notes", sa.Column("research_run_id", postgresql.UUID(as_uuid=False), nullable=True))
    op.add_column(
        "notes",
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    op.create_table(
        "conversation_papers",
        sa.Column("conversation_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("conversation_id", "paper_id"),
        sa.UniqueConstraint("conversation_id", "paper_id", name="uq_conversation_papers"),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="RESTRICT"),
    )
    op.create_index("ix_conversation_papers_paper_id", "conversation_papers", ["paper_id"])
    op.execute(
        """
        INSERT INTO conversation_papers (conversation_id, paper_id, position, created_at)
        SELECT id, paper_id, 0, created_at
        FROM conversations
        WHERE paper_id IS NOT NULL
        ON CONFLICT (conversation_id, paper_id) DO NOTHING
        """
    )

    op.create_table(
        "research_run_papers",
        sa.Column("research_run_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("research_run_id", "paper_id"),
        sa.UniqueConstraint("research_run_id", "paper_id", name="uq_research_run_papers"),
        sa.ForeignKeyConstraint(["research_run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="RESTRICT"),
    )
    op.create_index("ix_research_run_papers_paper_id", "research_run_papers", ["paper_id"])

    op.create_table(
        "evidence_records",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("research_run_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("chunk_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("evidence_id", sa.String(length=20), nullable=False),
        sa.Column("evidence_rank", sa.Integer(), nullable=False),
        sa.Column("similarity", sa.Float(), nullable=False),
        sa.Column("retrieval_method", sa.String(length=50), nullable=False, server_default="vector"),
        sa.Column("metadata_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("research_run_id", "evidence_id", name="uq_evidence_run_evidence_id"),
        sa.UniqueConstraint("research_run_id", "evidence_rank", name="uq_evidence_run_rank"),
        sa.UniqueConstraint("research_run_id", "chunk_id", name="uq_evidence_run_chunk"),
        sa.ForeignKeyConstraint(["research_run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="RESTRICT"),
    )
    op.create_index("ix_evidence_records_chunk_id", "evidence_records", ["chunk_id"])
    op.create_index("ix_evidence_records_paper_id", "evidence_records", ["paper_id"])

    op.create_table(
        "claim_evidence_links",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("research_run_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("evidence_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("citation_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("claim_start", sa.Integer(), nullable=True),
        sa.Column("claim_end", sa.Integer(), nullable=True),
        sa.Column("mapping_status", sa.String(length=20), nullable=False, server_default="unverified"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "mapping_status IN ('validated','unverified','rejected')",
            name="ck_claim_evidence_mapping_status",
        ),
        sa.ForeignKeyConstraint(["research_run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["evidence_id"], ["evidence_records.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["citation_id"], ["citations.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "human_checkpoint_decisions",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("research_run_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("trace_event_id", postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column("thread_id", sa.Text(), nullable=True),
        sa.Column("checkpoint_ns", sa.Text(), nullable=True),
        sa.Column("checkpoint_id", sa.Text(), nullable=True),
        sa.Column("decision", sa.String(length=30), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("reviewer", sa.String(length=255), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "decision IN ('approved','rejected','edited','abstained','resume_requested')",
            name="ck_human_checkpoint_decision",
        ),
        sa.ForeignKeyConstraint(["research_run_id"], ["research_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["trace_event_id"], ["trace_events.id"], ondelete="SET NULL"),
    )
    op.create_index(
        "ix_human_checkpoint_run_created",
        "human_checkpoint_decisions",
        ["research_run_id", "created_at"],
    )

    op.create_table(
        "indexing_runs",
        sa.Column("id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column("embedding_model", sa.String(length=255), nullable=False),
        sa.Column("embedding_dimension", sa.Integer(), nullable=False),
        sa.Column("chunker_version", sa.String(length=50), nullable=False, server_default="chunker-v1"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "status IN ('pending','running','completed','failed')",
            name="ck_indexing_runs_status",
        ),
        sa.ForeignKeyConstraint(["paper_id"], ["papers.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_indexing_runs_paper_started", "indexing_runs", ["paper_id", "started_at"])

    op.create_foreign_key(
        "papers_collection_id_fkey", "papers", "collections", ["collection_id"], ["id"], ondelete="SET NULL"
    )
    op.create_foreign_key(
        "chunks_paper_id_fkey", "chunks", "papers", ["paper_id"], ["id"], ondelete="CASCADE"
    )
    op.create_foreign_key(
        "research_runs_conversation_id_fkey",
        "research_runs",
        "conversations",
        ["conversation_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "research_runs_message_id_fkey",
        "research_runs",
        "messages",
        ["message_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "messages_research_run_id_fkey",
        "messages",
        "research_runs",
        ["research_run_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "trace_events_research_run_id_fkey",
        "trace_events",
        "research_runs",
        ["research_run_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "citations_research_run_id_fkey",
        "citations",
        "research_runs",
        ["research_run_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "citations_chunk_id_fkey",
        "citations",
        "chunks",
        ["chunk_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "citations_evidence_id_fkey",
        "citations",
        "evidence_records",
        ["evidence_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "feedback_research_run_id_fkey",
        "feedback",
        "research_runs",
        ["research_run_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "notes_paper_id_fkey", "notes", "papers", ["paper_id"], ["id"], ondelete="CASCADE"
    )
    op.create_foreign_key(
        "notes_conversation_id_fkey",
        "notes",
        "conversations",
        ["conversation_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "notes_research_run_id_fkey",
        "notes",
        "research_runs",
        ["research_run_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "conversations_paper_id_fkey",
        "conversations",
        "papers",
        ["paper_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "messages_conversation_id_fkey",
        "messages",
        "conversations",
        ["conversation_id"],
        ["id"],
        ondelete="CASCADE",
    )

    op.create_check_constraint(
        "ck_papers_index_status",
        "papers",
        "index_status IN ('pending','indexing','indexed','failed')",
    )
    op.create_check_constraint(
        "ck_chunks_vector_status",
        "chunks",
        "vector_status IN ('pending','indexed','failed','stale')",
    )
    op.create_check_constraint(
        "ck_notes_context",
        "notes",
        "paper_id IS NOT NULL OR conversation_id IS NOT NULL OR research_run_id IS NOT NULL",
    )


def downgrade() -> None:
    op.drop_table("indexing_runs")
    op.drop_table("human_checkpoint_decisions")
    op.drop_table("claim_evidence_links")
    op.drop_table("evidence_records")
    op.drop_table("research_run_papers")
    op.drop_table("conversation_papers")

    op.drop_constraint("ck_notes_context", "notes", type_="check")
    op.drop_column("notes", "updated_at")
    op.drop_column("notes", "research_run_id")
    op.drop_column("notes", "conversation_id")

    op.drop_constraint("uq_citations_run_index", "citations", type_="unique")
    op.drop_column("citations", "metadata_snapshot")
    op.drop_column("citations", "citation_text")
    op.drop_column("citations", "evidence_id")

    op.drop_constraint("uq_trace_events_run_sequence", "trace_events", type_="unique")
    op.drop_column("trace_events", "sequence")
    op.drop_column("trace_events", "thread_id")

    op.drop_column("messages", "research_run_id")
    op.drop_column("research_runs", "updated_at")
    op.drop_column("research_runs", "resume_token")
    op.drop_column("research_runs", "checkpoint_id")
    op.drop_column("research_runs", "thread_id")
    op.drop_column("research_runs", "top_k")
    op.drop_column("research_runs", "embedding_model")
    op.drop_column("research_runs", "model_name")
    op.drop_column("research_runs", "abstention_reason")
    op.drop_column("research_runs", "error")
    op.drop_column("research_runs", "answerable")
    op.drop_column("research_runs", "message_id")
    op.drop_column("research_runs", "conversation_id")

    op.drop_column("chunks", "updated_at")
    op.drop_column("chunks", "indexed_at")
    op.drop_column("chunks", "vector_id")
    op.drop_column("chunks", "vector_status")
    op.drop_column("chunks", "embedding_dimension")
    op.drop_column("chunks", "embedding_model")
    op.drop_column("chunks", "text_hash")

    op.drop_column("papers", "updated_at")
    op.drop_column("papers", "embedding_dimension")
    op.drop_column("papers", "embedding_model")
    op.drop_column("papers", "index_error")
    op.drop_column("papers", "index_status")
    op.drop_column("papers", "mime_type")
    op.drop_column("papers", "storage_key")
