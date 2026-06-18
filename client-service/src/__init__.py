"""
Initiates FastApi service
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from .config import Config
from .routes import initial_router

from .predictor import predictor_router
from .trainer import trainer_router

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

app.include_router(initial_router, tags=["initial_router"])
app.include_router(predictor_router, prefix=f"/api/{version}/predictor", tags=["predictor_router"])
app.include_router(trainer_router, prefix=f"/api/{version}/trainer", tags=["trainer_router"])

@app.get("/health")
def health_check():
    # Optional: Add database/Redis connectivity checks here
    return {"status": "healthy"}