from pydantic import BaseModel
from fastapi import UploadFile
from datetime import datetime, time
from typing import Optional, List

import uuid

class PredictSchema(BaseModel):
    data: List