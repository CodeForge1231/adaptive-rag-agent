from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routers import health
from src.bootstrap import bootstrap


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the application lifecycle and resource initialization.

    This context manager handles the startup process by bootstrapping 
    the application context and ensures a graceful shutdown by 
    releasing global resources.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.
    """
    # Initialize core services and dependencies
    ctx = await bootstrap()

    # Store the context in the app state for dependency injection
    app.state.context = ctx

    yield

    # Clean up resources
    await ctx.shutdown()


def create_app() -> FastAPI:
    """
    Configure and return the FastAPI application instance.

    This factory function initializes the app, sets metadata, 
    attaches the lifespan handler, and registers all API routers.

    Returns
    -------
    FastAPI
        A fully configured FastAPI application.
    """
    # Initialize the main FastAPI application
    app = FastAPI(title="RAG Orchestrator API", version="1.0.0", lifespan=lifespan)

    app.include_router(health.router)

    return app
