from types import SimpleNamespace

import pytest
from fastapi import FastAPI

import backend.app.main as main_module
import backend.app.checkpointer as checkpointer_module


def test_build_checkpointer_pool_configuration(monkeypatch):
    created = {}

    class FakePool:
        def __init__(self, **kwargs):
            created.update(kwargs)

    class FakeSaver:
        def __init__(self, conn, serde):
            created["conn"] = conn
            created["serde"] = serde
            created["setup_called"] = False

        def setup(self):
            created["setup_called"] = True

    monkeypatch.setattr(checkpointer_module, "ConnectionPool", FakePool)
    monkeypatch.setattr(checkpointer_module, "PostgresSaver", FakeSaver)

    resources = checkpointer_module.build_checkpointer(
        "postgresql://user:pass@localhost:5432/db"
    )

    assert created["conninfo"] == "postgresql://user:pass@localhost:5432/db"
    assert created["min_size"] == 1
    assert created["max_size"] == 5
    assert created["open"] is True
    assert created["kwargs"]["autocommit"] is True
    assert "row_factory" in created["kwargs"]
    assert created["setup_called"] is True
    assert isinstance(created["serde"], checkpointer_module.JsonPlusSerializer)
    assert isinstance(resources.saver, FakeSaver)


def test_close_checkpointer_closes_pool():
    closed = {"value": False}

    class FakePool:
        def close(self):
            closed["value"] = True

    resources = SimpleNamespace(pool=FakePool(), saver=object())
    checkpointer_module.close_checkpointer(resources)
    assert closed["value"] is True


def test_saver_construction_failure_closes_pool(monkeypatch):
    closed = {"value": False}

    class FakePool:
        def __init__(self, **kwargs):
            pass

        def close(self):
            closed["value"] = True

    class FailingSaver:
        def __init__(self, conn, serde):
            raise RuntimeError("saver construction failed")

    monkeypatch.setattr(checkpointer_module, "ConnectionPool", FakePool)
    monkeypatch.setattr(checkpointer_module, "PostgresSaver", FailingSaver)

    with pytest.raises(RuntimeError, match="saver construction failed"):
        checkpointer_module.build_checkpointer(
            "******localhost:5432/db"
        )

    assert closed["value"] is True


def test_saver_setup_failure_closes_pool(monkeypatch):
    closed = {"value": False}

    class FakePool:
        def __init__(self, **kwargs):
            pass

        def close(self):
            closed["value"] = True

    class FailingSetupSaver:
        def __init__(self, conn, serde):
            pass

        def setup(self):
            raise RuntimeError("saver setup failed")

    monkeypatch.setattr(checkpointer_module, "ConnectionPool", FakePool)
    monkeypatch.setattr(checkpointer_module, "PostgresSaver", FailingSetupSaver)

    with pytest.raises(RuntimeError, match="saver setup failed"):
        checkpointer_module.build_checkpointer(
            "******localhost:5432/db"
        )

    assert closed["value"] is True


def test_lifespan_initializes_and_closes(monkeypatch):
    app = FastAPI()
    lifecycle = {"started": False, "recovered": False, "closed": False}

    monkeypatch.setattr(main_module, "get_settings", lambda: SimpleNamespace(
        database_url="postgresql+psycopg://user:pass@localhost:5432/db"
    ))
    monkeypatch.setattr(
        main_module,
        "build_checkpointer",
        lambda url: lifecycle.update(started=True) or SimpleNamespace(saver=object(), pool=object()),
    )
    monkeypatch.setattr(
        main_module,
        "close_checkpointer",
        lambda resources: lifecycle.update(closed=True),
    )
    monkeypatch.setattr(
        main_module,
        "recover_pending_research_runs",
        lambda saver: lifecycle.update(recovered=True),
    )

    import asyncio

    async def run():
        async with main_module.lifespan(app):
            assert lifecycle["started"] is True
            assert lifecycle["recovered"] is True
            assert app.state.checkpointer_resources is not None

    asyncio.run(run())

    assert lifecycle["closed"] is True


def test_lifespan_startup_failure_propagates(monkeypatch):
    app = FastAPI()

    monkeypatch.setattr(main_module, "get_settings", lambda: SimpleNamespace(
        database_url="postgresql://user:pass@localhost:5432/db"
    ))
    monkeypatch.setattr(
        main_module,
        "build_checkpointer",
        lambda url: (_ for _ in ()).throw(RuntimeError("checkpointer setup failed")),
    )

    import asyncio

    async def run():
        async with main_module.lifespan(app):
            pass

    with pytest.raises(RuntimeError, match="checkpointer setup failed"):
        asyncio.run(run())


def test_health_endpoint_compatibility():
    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
