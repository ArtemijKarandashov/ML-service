from pydantic import BaseModel
from fastapi import UploadFile
from datetime import datetime, time
from typing import Optional, List

import uuid


class PredictFromCsvSchema(BaseModel):
    csv_file: UploadFile
    loan_status: Optional[str]

class PredictSchema(BaseModel):
    client_characteristics: List[float]