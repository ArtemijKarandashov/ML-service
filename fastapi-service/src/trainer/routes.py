from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import HTTPException

from .schemas import UploadFileSchema

trainer_router = APIRouter()
templates = Jinja2Templates(directory="templates/predictor")

@trainer_router.post("/upload-model", response_class=JSONResponse)
async def upload_model(
    data: UploadFileSchema
):
    """
    Accepts .pkl file and loads it into database.
    Returns uid of a file.
    """
    try:
        pkl_model = data.pkl_file.read()
        json_data = {
            "message": "File successfully uploaded"
        }

        return JSONResponse(content=json_data, status_code=200)
    
    except UnicodeDecodeError:
        return HTTPException(status_code=415, detail="Could not read .pkl file. Unsupported encoding provided?")
    except Exception as e:
        return HTTPException(status_code=500, detail="Unexpected exception occured upon reading .pkl file.")