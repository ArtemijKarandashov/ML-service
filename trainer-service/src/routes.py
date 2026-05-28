from fastapi import APIRouter, Request, UploadFile, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import HTTPException

from sqlmodel.ext.asyncio.session import AsyncSession

import os
import uuid

from .config import Config

trainer_router = APIRouter()