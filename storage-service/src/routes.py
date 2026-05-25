from fastapi import APIRouter, Request, UploadFile, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import HTTPException

from sqlmodel.ext.asyncio.session import AsyncSession

import os
import shutil
import uuid

from .config import Config
from .db import PklFileEntry, get_session
from .db.models import ModelStatus
from .service import PklFileService

storage_router = APIRouter()
templates = Jinja2Templates(directory="templates")


database_service = PklFileService()

@storage_router.get("/", response_class=HTMLResponse)
async def root(
    request: Request
):
    return {"message": "Storage service. Probably would show some data here later."}


@storage_router.get("/info/{uid}", response_class=JSONResponse)
async def get(
    uid: str,
    session: AsyncSession = Depends(get_session)
):
    try:
        file_entry: PklFileEntry = await database_service.get_pkl_file_entry(
            session=session,
            uid=uid
        )

        if file_entry:
            json_data = file_entry.model_dump_json()
            return JSONResponse(status_code=200, content=json_data)
        else:
            raise HTTPException(status_code=404, detail="PklFileEntry with specified uid does not exist.")

    except FileExistsError as e:
        return HTTPException(status_code=404, detail="Cannot find file with specified uid.")
    except PermissionError:
        return HTTPException(status_code=409, detail="Cannot read file. PermissionError.")
    except OSError:
        return HTTPException(status_code=500, detail="Unexpected error occured.")


@storage_router.get("/get/{uid}", response_class=FileResponse)
async def get(
    uid: str,
    session: AsyncSession = Depends(get_session)
):
    try:
        file_entry: PklFileEntry = await database_service.get_pkl_file_entry(
            session=session,
            uid=uid
        )

        if file_entry:
            return FileResponse(path=file_entry.filepath)
        else:
            raise HTTPException(status_code=404, detail="PklFileEntry with specified uid does not exist.")

    except FileExistsError as e:
        return HTTPException(status_code=404, detail="Cannot find file with specified uid.")
    except PermissionError:
        return HTTPException(status_code=409, detail="Cannot read file. PermissionError.")
    except OSError:
        return HTTPException(status_code=500, detail="Unexpected error occured.")


@storage_router.post("/store", response_class=JSONResponse)
async def store(
    pkl_file: UploadFile,
    session: AsyncSession = Depends(get_session)
):
    try:
        uid = str(uuid.uuid4())
        name = pkl_file.filename
        file_path = os.path.join(Config.UPLOAD_DIR, uid)
        status = ModelStatus.LOADED

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(pkl_file.file, buffer)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found. Failed to create .pkl file?")
        
        await database_service.create_pkl_file_entry(
            session=session,
            uid=uid,
            file_path=file_path,
            status=status,
            name=name
        )
        return JSONResponse(status_code=200, content={"message": "File saved", "uid": uid})
    
    except FileExistsError:
        return HTTPException(status_code=409, detail="File with set uid already exists.")
    except PermissionError:
        return HTTPException(status_code=409, detail="Cannot create file. PermissionError.")
    except OSError:
        return HTTPException(status_code=500, detail="Unexpected error occured.")


@storage_router.delete("/delete/{uid}", response_class=HTMLResponse)
async def delete(
    uid: str,
    session: AsyncSession = Depends(get_session)
):
    file: PklFileEntry = await database_service.get_pkl_file_entry(session=session, uid=uid)
    await database_service.delete_pkl_file_entry(session=session, uid=uid)
    os.remove(file.filepath)
    
    return JSONResponse(status_code=200, content={"message": "Deleted .pkl file", "uid": uid})