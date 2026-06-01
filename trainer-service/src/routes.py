from fastapi import APIRouter, Request, UploadFile, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.exceptions import HTTPException
from starlette.datastructures import Headers

from sqlmodel.ext.asyncio.session import AsyncSession
from typing import List
from pydantic import BaseModel

import io

from .config import Config
from .services import storage_service

trainer_router = APIRouter()

@trainer_router.post("/train")
async def train(
    csv_file: UploadFile
):
    try:
        # Scary? I dunno
        content = await csv_file.read()

        # do model call here with async. Resulting .pkl meant to be saved in file_bytes variable.
        filename = "PLACEHOLDER"
        file_bytes = b"PLACEHOLDER"
        file_stream = io.BytesIO(file_bytes)
        upload_file = UploadFile(
            file=file_stream, 
            filename=filename,
            headers=Headers({"content-type": "text/plain"}),
            size=len(file_bytes)
        )

        # saving .pkl model on storage server
        response = await storage_service.save_pkl_file(file=upload_file)
        response_data = response.json()
        json_data = {
            "message": "Model successfully trained",
            "uid": response_data["uid"]
        }
        return JSONResponse(content=json_data, status_code=200)
    except:
        pass

# FOR PREDICTOR

predictor_router = APIRouter()

@predictor_router.post("/predict")
async def predict(
    data: List[BaseModel]
):
    try:
        ...
    except:
        pass

@predictor_router.post("/predict_csv")
async def predict_csv(
    csv_file: UploadFile
):
    try:
        ...
    except:
        pass