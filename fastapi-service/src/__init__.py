"""
Initiates FastApi service
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from .config import Config
from .routes import initial_router

@asynccontextmanager
async def life_span(app: FastAPI):
    """Startup async functions are put here and are expected to be awaited"""
    print(f"API is starting...")
    # await example_startup()
    yield
    print(f"API has been stopped")

version = "v1"
app = FastAPI(
    title="ML-service",
    description="ML-service handler",
    version=version,
    lifespan=life_span
    )

templates = Jinja2Templates(directory="templates/")

app.mount("/static", StaticFiles(directory="static"), name="static")

# API ROUTERS
#app.include_router(example_router, prefix="/api/{version}/example", tags=["example"])

app.include_router(initial_router, prefix="/api/{version}/jinja2test", tags=["initial_router"])

