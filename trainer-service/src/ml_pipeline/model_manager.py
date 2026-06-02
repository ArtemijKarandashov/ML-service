import pickle
from io import BytesIO, TextIOWrapper
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


def save_model(model: object, path: str, model_name: str) -> None:
    with open(path + "/" + model_name, "wb") as file:
        pickle.dump(model, file)
        print(f"type of model: {type(model)}")
        print(f"Model {model_name} saved to {path}")


def load_model(path: str, model_name: str):
    with open(path + "/" + model_name, "rb") as file:
        loaded_model = pickle.load(file)

    return loaded_model


def load_model_from_bytes(model_bytes: bytes) -> object:
    model = pickle.loads(model_bytes)
    print(model)

    return model


def get_model_metrics(
    model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float, np.ndarray]:
    test_prediction = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, test_prediction),
        "roc_auc": roc_auc_score(y_test, test_prediction),
        "confusion_matrix": confusion_matrix(y_test, test_prediction),
    }

    return metrics


def predict_with_model(model: Any, data: pd.DataFrame) -> np.ndarray:
    prediction = model.predict(data)

    return prediction
