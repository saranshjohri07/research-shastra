import logging
from datetime import datetime, timezone
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command
from sqlalchemy import select

from ..agents.workflow import build_research_graph
from ..db.models import HumanCheckpointDecision, ResearchRun
from ..db.session import SessionLocal


logger = logging.getLogger(__name__)


class ResearchRecoveryError(RuntimeError):
    pass


def recover_pending_research_runs(checkpointer: BaseCheckpointSaver) -> None:
    with SessionLocal() as db:
        run_ids = db.scalars(
            select(ResearchRun.id)
            .where(
                ResearchRun.status.in_(("running", "failed")),
                ResearchRun.resume_token.is_not(None),
            )
            .order_by(ResearchRun.created_at)
        ).all()

    for run_id in run_ids:
        try:
            recovered = _recover_run(run_id, checkpointer)
        except Exception as error:
            logger.exception("Unable to recover research run %s", run_id)
            _record_recovery_error(run_id, error)
            continue
        if recovered:
            logger.info("Recovered research run %s", run_id)


def _recover_run(run_id: str, checkpointer: BaseCheckpointSaver) -> bool:
    with SessionLocal.begin() as db:
        run = db.scalar(
            select(ResearchRun)
            .where(
                ResearchRun.id == run_id,
                ResearchRun.status.in_(("running", "failed")),
                ResearchRun.resume_token.is_not(None),
            )
            .with_for_update(skip_locked=True)
        )
        if run is None:
            return False

        decisions = db.scalars(
            select(HumanCheckpointDecision)
            .where(HumanCheckpointDecision.research_run_id == run.id)
            .order_by(HumanCheckpointDecision.created_at)
        ).all()
        if len(decisions) != 1:
            raise ResearchRecoveryError(
                f"Expected one persisted human decision, found {len(decisions)}"
            )
        decision = decisions[0]
        thread_id = run.thread_id
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ResearchRecoveryError(
                "Research run has no valid persisted thread_id"
            )
        try:
            UUID(thread_id)
        except ValueError as error:
            raise ResearchRecoveryError(
                "Research run has no valid persisted thread_id"
            ) from error
        if decision.thread_id != thread_id:
            raise ResearchRecoveryError(
                "Persisted decision thread_id does not match the research run"
            )
        if decision.decision not in ("approved", "rejected"):
            raise ResearchRecoveryError("Persisted decision is not resumable")
        if not decision.checkpoint_id:
            raise ResearchRecoveryError(
                "Persisted decision has no checkpoint_id"
            )

        config = {"configurable": {"thread_id": thread_id}}
        graph = build_research_graph(db, checkpointer=checkpointer)
        snapshot = graph.get_state(config)
        if snapshot.values.get("thread_id") != thread_id:
            raise ResearchRecoveryError(
                "Saved checkpoint identity does not match the research run"
            )

        _verify_decision_checkpoint(graph, config, decision, thread_id)
        expected_user_decision = (
            "continue" if decision.decision == "approved" else "flag"
        )
        pending_human_review = any(
            isinstance(getattr(interrupt, "value", None), dict)
            and interrupt.value.get("type") == "human_review"
            for interrupt in snapshot.interrupts
        )
        if snapshot.interrupts and not pending_human_review:
            raise ResearchRecoveryError(
                "Saved checkpoint has an unexpected pending interrupt"
            )

        if pending_human_review:
            if not _snapshot_matches_decision_checkpoint(snapshot, decision):
                raise ResearchRecoveryError(
                    "Pending human review does not match the persisted decision checkpoint"
                )
            result = dict(
                graph.invoke(
                    Command(
                        resume={
                            "decision": decision.decision,
                            "feedback": decision.comment,
                        }
                    ),
                    config=config,
                )
            )
        elif snapshot.values.get("user_decision") == expected_user_decision:
            if snapshot.next:
                result = dict(graph.invoke(None, config=config))
            else:
                result = dict(snapshot.values)
        else:
            raise ResearchRecoveryError(
                "Saved checkpoint does not contain the persisted human decision"
            )

        if result.get("__interrupt__"):
            raise ResearchRecoveryError(
                "Graph recovery returned with an unresolved interrupt"
            )
        if result.get("thread_id") != thread_id:
            raise ResearchRecoveryError(
                "Recovered graph result thread_id does not match the research run"
            )
        if result.get("user_decision") != expected_user_decision:
            raise ResearchRecoveryError(
                "Recovered graph result does not match the persisted decision"
            )
        if result.get("status") not in (
            "completed",
            "abstained",
            "failed",
            "flagged",
        ):
            raise ResearchRecoveryError(
                "Recovered graph has no pending tasks and no terminal status"
            )
        if decision.decision == "rejected" and result.get("status") != "flagged":
            raise ResearchRecoveryError(
                "Rejected decision did not follow the graph flag branch"
            )
        if decision.decision == "approved" and result.get("status") == "flagged":
            raise ResearchRecoveryError(
                "Approved decision incorrectly followed the graph flag branch"
            )

        persist_graph_result(run, result)
    return True


def _verify_decision_checkpoint(
    graph,
    config: dict,
    decision: HumanCheckpointDecision,
    thread_id: str,
) -> None:
    for checkpoint in graph.get_state_history(config, limit=1000):
        checkpoint_config = getattr(checkpoint, "config", None) or {}
        configurable = checkpoint_config.get("configurable", {})
        if (
            configurable.get("thread_id") == thread_id
            and configurable.get("checkpoint_ns", "") == (decision.checkpoint_ns or "")
            and configurable.get("checkpoint_id") == decision.checkpoint_id
        ):
            if checkpoint.values.get("thread_id") == thread_id:
                return
    raise ResearchRecoveryError(
        "Persisted decision checkpoint is missing or does not match the run"
    )


def _snapshot_matches_decision_checkpoint(
    snapshot,
    decision: HumanCheckpointDecision,
) -> bool:
    checkpoint_config = getattr(snapshot, "config", None) or {}
    configurable = checkpoint_config.get("configurable", {})
    return (
        configurable.get("checkpoint_ns", "") == (decision.checkpoint_ns or "")
        and configurable.get("checkpoint_id") == decision.checkpoint_id
    )


def persist_graph_result(run: ResearchRun, result: dict) -> None:
    run.status = {
        "completed": "completed",
        "abstained": "abstained",
        "failed": "failed",
        "flagged": "cancelled",
    }.get(result.get("status"), "running")
    run.answer = result.get("draft_answer")
    run.answerable = result.get("answerable")
    run.error = result.get("error")
    run.abstention_reason = result.get("abstention_reason")
    if run.status in {"completed", "abstained", "failed", "cancelled"}:
        run.completed_at = datetime.now(timezone.utc)


def _record_recovery_error(run_id: str, error: Exception) -> None:
    try:
        with SessionLocal.begin() as db:
            run = db.scalar(
                select(ResearchRun)
                .where(
                    ResearchRun.id == run_id,
                    ResearchRun.status.in_(("running", "failed")),
                    ResearchRun.resume_token.is_not(None),
                )
                .with_for_update(skip_locked=True)
            )
            if run is not None:
                run.error = f"Research run recovery pending: {error}"
    except Exception:
        logger.exception("Unable to persist recovery error for research run %s", run_id)
