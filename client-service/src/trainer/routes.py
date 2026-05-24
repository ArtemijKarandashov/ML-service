from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import HTTPException

from src.trainer.schemas import UploadFileModel
from src.config import Config
from src.services import storage_service

trainer_router = APIRouter()
templates = Jinja2Templates(directory="templates/predictor")

@trainer_router.post("/upload-model", response_class=JSONResponse)
async def upload_model(
    file: UploadFile
):
    """
    Accepts .pkl file and loads it into database.
    Returns uid of a file.
    """
    try:
        response = await storage_service.save_pkl_file(file=file)
        if not response:
            raise HTTPException(status_code=409, detail="Could not save file.")
        response_data = response.json()
        json_data = {
            "message": response_data["message"],
            "uid": response_data["uid"]
        }

        return JSONResponse(content=json_data, status_code=200)
    
    except UnicodeDecodeError:
        return HTTPException(status_code=415, detail="Could not read .pkl file. Unsupported encoding provided?")


@trainer_router.get("/get-model/{uid}")
async def get_model(
    uid: str
):
    """
    Gets model with given uid from Storage service. 
    """
    try:
        response = await storage_service.get_pkl_file(uid=uid)
        if not response:
            raise HTTPException(status_code=409, detail="Could not get file.")

        return Response(
            content=response.content,
            media_type="application/octet-stream"
        )
    
    except UnicodeDecodeError:
        return HTTPException(status_code=415, detail="Could not get .pkl file. Unsupported encoding provided?")


@trainer_router.delete("/delete-model/{uid}", response_class=JSONResponse)
async def delete_model(
    uid: str
):
    """
    Accepts .pkl file and loads it into database.
    Returns uid of a file.
    """
    try:
        response = await storage_service.delete_pkl_file(uid=uid)
        if not response:
            raise HTTPException(status_code=409, detail="Could not delete file.")
        response_data = response.json()
        json_data = {
            "message": response_data["message"],
            "uid": response_data["uid"]
        }

        return JSONResponse(content=json_data, status_code=200)
    
    except UnicodeDecodeError:
        return HTTPException(status_code=415, detail="Could not read .pkl file. Unsupported encoding provided?")