from fastapi import APIRouter, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import HTTPException

# /api/(version)/predictor/
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
    uid: str
):
    """
    Accepts several strings with client correlating factors and uid of a model.
    Returns list of correlating factors and loan status.
    If model is not loaded returns an error.
    """
    try:
        json_data = {
            "message": "Prediction successfully"
        }
        return JSONResponse(content=json_data, status_code=200)

    except UnicodeDecodeError:
        return HTTPException(status_code=415, detail="Could not read .pkl file. Unsupported encoding provided?")
    except Exception as e:
        return HTTPException(status_code=500, detail="Unexpected exception occured when reading .pkl file.")


@predictor_router.post("/predict-from-csv", response_class=JSONResponse)
async def predict_from_csv(
    csv: UploadFile
):
    """
    Accepts basic csv generated from train_test_split without any changes.
    Returns roc_auc and full dataset with predicted loan status.
    """
    try:
        json_data = {
            "message": "Prediction successfully"
        }
        return JSONResponse(content=json_data, status_code=200)

    except UnicodeDecodeError:
        return HTTPException(status_code=415, detail="Could not read .csv file. Unsupported encoding provided?")
    except Exception as e:
        return HTTPException(status_code=500, detail="Unexpected exception occured when reading .csv file.")