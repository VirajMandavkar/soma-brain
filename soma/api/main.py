"""
SOMA Brain — FastAPI Application Factory

Why a factory function instead of a module-level `app = FastAPI()`?
Testing. With a factory, each test can create an isolated app instance
with different settings (e.g., test database URL). Module-level singletons
make test isolation painful.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from soma.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks. Database pools, cache warmup, etc."""
    settings = get_settings()
    logger.info("SOMA Brain starting — log_level={}", settings.log_level)
    yield
    logger.info("SOMA Brain shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="SOMA Brain",
        description="Synthetic Organismic Modelling and Analysis — Brain. "
                    "Enterprise CNS drug discovery platform.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — permissive for local dev, lock down in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],  # Next.js dev server
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register route modules
    from soma.api.routes import simulation, patient, knowledge, literature
    app.include_router(simulation.router, prefix="/api/v1/simulation", tags=["Simulation"])
    app.include_router(patient.router, prefix="/api/v1/patient", tags=["Patient"])
    app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["Knowledge Graph"])
    app.include_router(literature.router, prefix="/api/v1/literature", tags=["Literature"])

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "version": "0.1.0"}

    return app


# Module-level instance for `uvicorn soma.api.main:app`
app = create_app()
