"""
Initiates FastApi service
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import Config
from .routes import trainer_router

# async def ensure_dirs():
#     if not os.path.exists("data/uploads"):
#         os.makedirs("data/uploads/", exist_ok=True)


@asynccontextmanager
async def life_span(app: FastAPI):
    """Startup async functions are put here and are expected to be awaited"""
    print(f"Trainer-API is starting...")
    # await example_startup()
    yield
    print(f"Trainer-API has been stopped")


version = "v1"
app = FastAPI(
    title="Trainer-service",
    description="Trainer-service handler",
    version=version,
    lifespan=life_span,
)
# Specify cors through env variable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,  # Allows cookies and auth headers
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# templates = Jinja2Templates(directory="templates/")
# app.mount("/static", StaticFiles(directory="static"), name="static")

# API ROUTERS
# app.include_router(example_router, prefix="/api/{version}/example", tags=["example"])

app.include_router(
    trainer_router, prefix=f"/api/{version}/trainer", tags=["trainer_router"]
)
