from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .errors import unhandled_exception_handler
from .logging_config import configure_logging
from .routers.conversations import router as conversations_router
from .routers.health import router as health_router
from .routers.papers import router as papers_router
from .routers.research import router as research_router
from .checkpointer import build_checkpointer, close_checkpointer
from .services.research_recovery import recover_pending_research_runs

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    database_url = settings.database_url
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    elif database_url.startswith("postgresql+psycopg://"):
        database_url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)

    checkpointer_resources = None
    try:
        checkpointer_resources = build_checkpointer(database_url)
        app.state.checkpointer_resources = checkpointer_resources
        recover_pending_research_runs(checkpointer_resources.saver)
        yield
    finally:
        if checkpointer_resources is not None:
            close_checkpointer(checkpointer_resources)


app = FastAPI(
    title="ResearchShastra API",
    version="0.1.0",
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(health_router)
app.include_router(papers_router)
app.include_router(research_router)
app.include_router(conversations_router)
