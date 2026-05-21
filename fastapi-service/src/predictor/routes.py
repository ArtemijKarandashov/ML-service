from fastapi import APIRouter, Request, jsonify
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

# /api/(version)/predict/
predictor_router = APIRouter()
templates = Jinja2Templates(directory="templates/predictor")

@predictor_router.get("/", response_class=JSONResponse)
async def root(
    request: Request
):
    json_data = {"message": "Predictor API functions are available on this route"}
    return JSONResponse(content=json_data, status_code=200)


@predictor_router.post("/predict", response_class=JSONResponse)
async def predict(
    request: Request
):
    """
    Accepts several strings with client correlating factors.
    Returns list of correlating factors and loan status.
    If model is not loaded returns an error.
    """
    json_data = {"message": "Predictor API functions are available on this route"}
    return JSONResponse(content=json_data, status_code=200)

@predictor_router.post("/predict-from-csv", response_class=JSONResponse)
async def predict(
    request: Request
):
    """
    Accepts basic csv generated from train_test_split without any changes.
    Returns ro_auc and full dataset with predicted loan status.
    """
    json_data = {"message": "Predictor API functions are available on this route"}
    return JSONResponse(content=json_data, status_code=200)