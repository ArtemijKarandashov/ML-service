import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)


def save_model(model: object, path: str, model_name: str) -> None:
    """
    Save a trained machine learning model to disk.

    Args:
        model (object): The machine learning model to be saved.
        path (str): The directory path where the model will be saved.
        model_name (str): The name of the file for saving the model.

    Returns:
        None

    """
    with open(path + "/" + model_name, "wb") as file:
        pickle.dump(model, file)
        print(f"Model {model_name} saved to {path}")


def save_model_bytes(model: object) -> bytes:
    """
    Save a machine learning model to bytes format.

    Args:
        model (object): The machine learning model to serialize.

    Returns:
        bytes: The serialized machine learning model in bytes format.

    Raises:
        PickleError: If the model cannot be serialized due to unsupported types.

    """
    model_bytes = pickle.dumps(model)

    return model_bytes


def load_model(path: str, model_name: str) -> object:
    """
    Load a model from disk.

    Args:
        path (str): The directory path where the model is stored.
        model_name (str): The name of the file containing the model.

    Returns:
        object: The loaded model.

    Raises:
        FileNotFoundError: If the specified file does not exist or cannot be accessed.
    """
    with open(path + "/" + model_name, "rb") as file:
        loaded_model = pickle.load(file)

    return loaded_model


def load_model_from_bytes(model_bytes: bytes) -> object:
    """
    Load a machine learning model from bytes.

    Args:
        model_bytes (bytes): The serialized machine learning model in bytes format.

    Returns:
        object: The loaded machine learning model.

    """
    model = pickle.loads(model_bytes)

    return model


def get_model_metrics(
    model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float, np.ndarray]:
    """
    Calculate and retrieve various performance metrics for a given machine learning model.

    Args:
        model (object): The trained machine learning model to use for predictions.
        x_test (pd.DataFrame): The feature matrix of the test set.
        y_test (pd.Series): The true labels corresponding to x_test.

    Returns:
        dict[str, Any]: A dictionary containing the following keys:
            - "accuracy" (float): The overall accuracy score.
            - "roc_auc" (float): The Area Under the Receiver Operating Characteristic Curve (AUC-ROC).
            - "confusion_matrix" (np.ndarray): A 2D array representing the confusion matrix.

    Raises:
        ValueError: If the model's predictions are incompatible with the expected output for accuracy_score or roc_auc_score calculations.
        AttributeError: If the provided x_test or y_test do not have the expected attributes.

    """
    test_prediction = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, test_prediction),
        "roc_auc": roc_auc_score(y_test, test_prediction),
        "confusion_matrix": confusion_matrix(y_test, test_prediction),
    }

    return metrics


def predict_with_model(
    model: Any,
    data: pd.DataFrame,
) -> np.ndarray:
    """
    Predicts the output for a given machine learning model and input data.

    Args:
        model (Any): The trained machine learning model to use for predictions.
        data (pd.DataFrame): The feature matrix of the test set corresponding to which you want to make predictions.

    Returns:
        np.ndarray: A numpy array containing the predicted values.

    Raises:
        ValueError: If the model's prediction method is incompatible with the input data type or shape.
        AttributeError: If the provided x_test or y_test do not have the expected attributes.

    """
    prediction = model.predict(data)

    return prediction
