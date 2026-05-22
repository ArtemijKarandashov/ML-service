from pydantic import BaseModel
from fastapi import UploadFile
from datetime import datetime, time
from typing import Optional

import uuid


class UploadFileModel(BaseModel):
    pkl_file: UploadFile