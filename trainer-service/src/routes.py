from fastapi import APIRouter, Request, UploadFile, Depends
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.exceptions import HTTPException
from starlette.datastructures import Headers

from .ml_pipeline.pipeline_manager import (
    PredictionPipeline,
    TrainingPipeline,
)
from .schemas import PredictSchema

from typing import List
from pydantic import BaseModel

import io
import csv
import asyncio

from .config import Config
from .services import storage_service

trainer_router = APIRouter()

@trainer_router.post("/train")
async def train(
    csv_file: UploadFile
):
    try:
        # Scary? I dunno
        csv_bytes = await csv_file.read()

        trainer = TrainingPipeline(csv_bytes)
        trainer.run_pipeline()

        model_name = csv_file.filename
        model_bytes = await asyncio.to_thread(trainer.get_model_bytes)
        metrics = trainer.get_metrics()

        file_stream = io.BytesIO(model_bytes)
        upload_file = UploadFile(
            file=file_stream, 
            filename=model_name,
            headers=Headers({"content-type": "text/plain"}),
            size=len(model_bytes)
        )

        # saving .pkl model on storage server
        response = await storage_service.save_pkl_file(file=upload_file)
        response_data = response.json()
        print(response_data)
        json_data = {
            "message": "Model successfully trained",
            "uid": response_data["uid"]
        }
        return JSONResponse(content=json_data, status_code=200)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not train model")

# FOR PREDICTOR

predictor_router = APIRouter()

csv_fieldnames = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
    "loan_status"
    ]

@predictor_router.post("/predict/{uid}")
async def predict(
    uid: str,
    PredictSchema: PredictSchema
):
    try:
        data = PredictSchema.data
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        print(buffer.getvalue().encode('utf-8'))
        csv_bytes = buffer.getvalue().encode('utf-8')

        model =  await storage_service.get_pkl_file(uid=uid)
        model_bytes = model.content

        prediction_pipeline = PredictionPipeline(csv_bytes, model_bytes)
        prediction_pipeline.run_pipeline()

        prediction = prediction_pipeline.get_prediction()
        response = {
            "message": "Successeful prediction",
            "predicion": prediction
        }
        return JSONResponse(content=response, status_code=200)
    except:
        raise HTTPException(status_code=500, detail="Unecpected error")

@predictor_router.post("/predict_csv/{uid}")
async def predict_csv(
    uid: str,
    csv_file: UploadFile
):
    try:
        csv_bytes = await csv_file.read()
        model =  await storage_service.get_pkl_file(uid=uid)
        model_bytes = model.content

        prediction_pipeline = PredictionPipeline(csv_bytes, model_bytes)
        prediction_pipeline.run_pipeline()

        prediction = prediction_pipeline.get_prediction()
        response = {
            "message": "Successeful prediction",
            "predicion": prediction.tolist()
        }
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Unecpected error")
