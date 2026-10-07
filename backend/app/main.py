from fastapi import FastAPI

from .errors import unhandled_exception_handler
from .logging_config import configure_logging
from .routers.health import router as health_router
from .routers.papers import router as papers_router
from .routers.research import router as research_router

configure_logging()

app = FastAPI(
    title="ResearchShastra API",
    version="0.1.0",
)

app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(health_router)
app.include_router(papers_router)
app.include_router(research_router)
