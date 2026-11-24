import json
import os
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

import psycopg
import pytest
from dotenv import dotenv_values
from psycopg import sql
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from backend.app.db.base import Base
from backend.app.db import models  # noqa: F401
from backend.app.db.models import Paper, ResearchRun


TEST_DATABASE_URL_ENV = "RESEARCHSHASTRA_RESTART_TEST_DATABASE_URL"
APP_BOOTSTRAP = r"""
import signal
import sys
import os

import psycopg
from sqlalchemy.engine import make_url

from backend.app.agents import stages
from backend.app.agents import workflow
from backend.app.retrieval.retriever import RetrievedEvidence


def retrieve_test_evidence(db, paper_id, question, top_k):
    return [
        RetrievedEvidence(
            evidence_id="E1",
            chunk_id="00000000-0000-4000-8000-000000000001",
            paper_id=paper_id,
            text="Deterministic process-restart evidence.",
            chunk_index=0,
            page_start=1,
            page_end=1,
            section="Methods",
            title="Process restart fixture",
            filename="process-restart.pdf",
            similarity=0.99,
        )
    ]


def generate_test_answer(question, context):
    return {
        "answerable": True,
        "answer": "The fixture supports this answer [E1].",
        "reason": None,
    }


stages.retrieve_evidence = retrieve_test_evidence
stages.generate_grounded_answer = generate_test_answer

for stage_name in ("reasoner", "citation_agent", "validator"):
    original_stage = getattr(workflow, stage_name)

    def count_stage_execution(state, original=original_stage, name=stage_name):
        database_url = make_url(os.environ["DATABASE_URL"])
        connection_parameters = database_url.translate_connect_args()
        connection_parameters["dbname"] = connection_parameters.pop("database")
        connection_parameters["user"] = connection_parameters.pop("username")
        connection_parameters.update(database_url.query)
        with psycopg.connect(**connection_parameters) as connection:
            connection.execute(
                "INSERT INTO restart_stage_counts (stage, executions) "
                "VALUES (%s, 1) "
                "ON CONFLICT (stage) "
                "DO UPDATE SET executions = "
                "restart_stage_counts.executions + 1",
                (name,),
            )
        return original(state)

    setattr(workflow, stage_name, count_stage_execution)

from backend.app.main import app
import uvicorn

server = uvicorn.Server(
    uvicorn.Config(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="error")
)


def request_shutdown(signum, frame):
    server.should_exit = True


signal.signal(signal.SIGTERM, request_shutdown)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, request_shutdown)
server.run()
"""


def isolated_schema_url(base_url: str, schema: str) -> str:
    url = make_url(base_url)
    if url.drivername == "postgresql":
        url = url.set(drivername="postgresql+psycopg")
    return url.update_query_dict(
        {"options": f"-csearch_path={schema}"}
    ).render_as_string(hide_password=False)


def start_application(database_url: str, port: int) -> subprocess.Popen:
    environment = os.environ.copy()
    environment.update(
        {
            "DATABASE_URL": database_url,
            "GROQ_API_KEY": "process-restart-integration-test",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return subprocess.Popen(
        [sys.executable, "-c", APP_BOOTSTRAP, str(port)],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP
            if os.name == "nt"
            else 0
        ),
    )


def stop_application(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        output = process.stdout.read() if process.stdout else ""
        raise AssertionError(f"Application exited unexpectedly:\n{output}")

    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.send_signal(signal.SIGTERM)
    try:
        return_code = process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
        output = process.stdout.read() if process.stdout else ""
        raise AssertionError(
            f"Application did not shut down gracefully:\n{output}"
        )

    output = process.stdout.read() if process.stdout else ""
    if return_code != 0:
        raise AssertionError(
            f"Application shutdown returned {return_code}:\n{output}"
        )


def wait_for_application(process: subprocess.Popen, url: str) -> None:
    deadline = time.monotonic() + 30
    last_error = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise AssertionError(f"Application failed to start:\n{output}")
        try:
            with urlopen(f"{url}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, URLError) as error:
            last_error = error
        time.sleep(0.2)
    raise AssertionError(f"Application did not become ready: {last_error}")


def post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=15) as response:
        return json.load(response)


def post_json_with_status(url: str, payload: dict) -> tuple[int, dict]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def free_local_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def test_http_resume_survives_full_application_process_restart():
    base_url = os.environ.get(TEST_DATABASE_URL_ENV)
    if not base_url:
        pytest.skip(
            f"Set {TEST_DATABASE_URL_ENV} to a disposable PostgreSQL test database"
        )

    database_name = urlsplit(base_url).path.rsplit("/", 1)[-1].split("?", 1)[0]
    if "test" not in database_name.lower():
        pytest.fail(
            f"{TEST_DATABASE_URL_ENV} must name a dedicated test database"
        )
    repository_root = Path(__file__).resolve().parents[2]
    configured_database_url = (
        os.environ.get("DATABASE_URL")
        or dotenv_values(repository_root / ".env").get("DATABASE_URL")
        or (
            "postgresql+psycopg://researchshastra:researchshastra_dev"
            "@127.0.0.1:5432/researchshastra"
        )
    )
    if make_url(base_url).database == make_url(configured_database_url).database:
        pytest.fail(
            f"{TEST_DATABASE_URL_ENV} must not target the configured application database"
        )

    schema = f"rs_restart_test_{uuid4().hex}"
    schema_url = isolated_schema_url(base_url, schema)
    admin_url = make_url(base_url).set(drivername="postgresql").render_as_string(
        hide_password=False
    )
    with psycopg.connect(admin_url, autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
        )

    engine = create_engine(schema_url, pool_pre_ping=True)
    processes = []
    try:
        Base.metadata.create_all(engine)
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE restart_stage_counts ("
                    "stage TEXT PRIMARY KEY, executions INTEGER NOT NULL)"
                )
            )
        paper_id = str(uuid4())
        with Session(engine) as db:
            db.add(
                Paper(
                    id=paper_id,
                    title="Process restart fixture",
                    filename="process-restart.pdf",
                    file_path="process-restart.pdf",
                    page_count=1,
                )
            )
            db.commit()

        first_port = free_local_port()
        first_url = f"http://127.0.0.1:{first_port}"
        first_process = start_application(schema_url, first_port)
        processes.append(first_process)
        wait_for_application(first_process, first_url)
        start_responses = [
            post_json(
                f"{first_url}/research/graph/query",
                {"paper_ids": [paper_id], "question": "What supports the answer?"},
            )
            for _ in range(2)
        ]
        first_response, second_response = start_responses
        assert first_response["status"] == "awaiting_review"
        assert second_response["status"] == "awaiting_review"
        first_run_id = first_response["research_run_id"]
        first_thread_id = first_response["thread_id"]
        second_run_id = second_response["research_run_id"]
        second_thread_id = second_response["thread_id"]
        assert first_run_id != second_run_id
        assert first_thread_id != second_thread_id

        with Session(engine) as db:
            for run_id, thread_id in (
                (first_run_id, first_thread_id),
                (second_run_id, second_thread_id),
            ):
                persisted_run = db.get(ResearchRun, run_id)
                assert persisted_run is not None
                assert persisted_run.thread_id == thread_id
                assert persisted_run.status == "running"
        stop_application(first_process)

        second_port = free_local_port()
        second_url = f"http://127.0.0.1:{second_port}"
        second_process = start_application(schema_url, second_port)
        processes.append(second_process)
        wait_for_application(second_process, second_url)

        third_port = free_local_port()
        third_url = f"http://127.0.0.1:{third_port}"
        third_process = start_application(schema_url, third_port)
        processes.append(third_process)
        wait_for_application(third_process, third_url)

        def resume(base_url, run_id):
            return post_json_with_status(
                f"{base_url}/research/runs/{run_id}/resume",
                {"decision": "approved", "feedback": "Reviewed after restart"},
            )

        with ThreadPoolExecutor(max_workers=3) as executor:
            first_attempt = executor.submit(resume, second_url, first_run_id)
            duplicate_attempt = executor.submit(resume, third_url, first_run_id)
            independent_attempt = executor.submit(resume, second_url, second_run_id)
            first_result = first_attempt.result()
            duplicate_result = duplicate_attempt.result()
            independent_result = independent_attempt.result()

        same_run_results = [first_result, duplicate_result]
        assert sorted(result[0] for result in same_run_results) == [200, 409], (
            f"Unexpected concurrent resume responses: {same_run_results!r}"
        )
        successful_same_run = next(
            body for response_status, body in same_run_results
            if response_status == 200
        )
        conflict_same_run = next(
            body for response_status, body in same_run_results
            if response_status == 409
        )
        assert conflict_same_run["detail"] == (
            "Research run resume has already been claimed or is not resumable"
        )
        assert independent_result[0] == 200
        assert independent_result[1]["status"] == "completed"
        assert successful_same_run["status"] == "completed"
        assert successful_same_run["research_run_id"] == first_run_id
        assert successful_same_run["thread_id"] == first_thread_id
        assert successful_same_run["draft_answer"] == (
            "The fixture supports this answer [E1]."
        )
        assert independent_result[1]["research_run_id"] == second_run_id
        assert independent_result[1]["thread_id"] == second_thread_id

        with Session(engine) as db:
            for run_id, thread_id in (
                (first_run_id, first_thread_id),
                (second_run_id, second_thread_id),
            ):
                completed_run = db.get(ResearchRun, run_id)
                assert completed_run is not None
                assert completed_run.status == "completed"
                assert completed_run.thread_id == thread_id
                assert completed_run.answer == (
                    "The fixture supports this answer [E1]."
                )
            counts = dict(
                db.execute(
                    text("SELECT stage, executions FROM restart_stage_counts")
                ).all()
            )
            decision_rows = db.execute(
                text(
                    "SELECT research_run_id::text, thread_id, decision, comment "
                    "FROM human_checkpoint_decisions"
                )
            ).all()
        assert counts == {
            "reasoner": 2,
            "citation_agent": 2,
            "validator": 2,
        }
        assert len(decision_rows) == 2
        assert {
            run_id: (stored_thread_id, decision, comment)
            for run_id, stored_thread_id, decision, comment in decision_rows
        } == {
            first_run_id: (
                first_thread_id,
                "approved",
                "Reviewed after restart",
            ),
            second_run_id: (
                second_thread_id,
                "approved",
                "Reviewed after restart",
            ),
        }
    finally:
        shutdown_error = None
        try:
            for process in reversed(processes):
                if process.poll() is None:
                    stop_application(process)
        except AssertionError as error:
            shutdown_error = error
        finally:
            engine.dispose()
            with psycopg.connect(admin_url, autocommit=True) as connection:
                connection.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(schema)
                    )
                )
        if shutdown_error is not None:
            raise shutdown_error
