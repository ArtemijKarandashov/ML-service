import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_dataframe(csv_path: str, csv_name: str) -> pd.DataFrame:
    # TODO:
    # checks:
    #   - file exists?
    #   - is it csv?
    df = pd.read_csv(csv_path + "/" + csv_name)

    return df


def encode_binary(df: pd.DataFrame, column: str, target_value: str) -> pd.DataFrame:
    """
    Encodes a binary categorical feature into an integer value.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to be encoded.
        target_value (str): The target value for the encoding.

    Returns:
        pd.DataFrame: A new DataFrame with the encoded column appended.

    """
    df_encoded = df[column].apply(
        lambda value: 1 if value.lower() == target_value else 0,
    )
    df_encoded.rename(str(column + "_encoded"), inplace=True)

    return df_encoded


def encode_one_hot(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df_encoded = (
        pd.get_dummies(
            df[column].apply(pd.Series).stack(),
        )
        .groupby(level=0)
        .sum()
    )

    return df_encoded


# Dictionary with the encoding strategies for categorical features
encoding_strategy = {
    "person_gender": lambda df, column: encode_binary(
        df,
        column,
        "male",
    ),
    "previous_loan_defaults_on_file": lambda df, column: encode_binary(
        df,
        column,
        "yes",
    ),
    "person_education": lambda df, column: encode_one_hot(df, column),
    "person_home_ownership": lambda df, column: encode_one_hot(df, column),
    "loan_intent": lambda df, column: encode_one_hot(df, column),
}


def encode_columns(df_sanitized: pd.DataFrame, encoding_strategy: dict) -> pd.DataFrame:
    df_encoded = df_sanitized.copy()

    for column in df_encoded:
        if column in encoding_strategy:
            print(f"Encoding column {column}")
            encoded_column = encoding_strategy[column](df_encoded, column)
            df_encoded = pd.concat(
                [df_encoded.drop(column, axis=1), encoded_column],
                axis=1,
            )
    # TODO:
    #  - drop unsupported columns

    print("Encoding completed")

    return df_encoded


def drop_low_corr_columns(
    df: pd.DataFrame,
    target_column: str,
    drop_threshold: float = 0.004,
) -> pd.DataFrame:
    df_cleaned = df.copy()
    correlation_series = df.corrwith(df[target_column]).abs()
    for column in df.columns:
        if correlation_series[column] < drop_threshold:
            print(
                f"Dropper column {column} with correlation {correlation_series[column]}",
            )
            df_cleaned.drop(column, axis=1)

    return df_cleaned


def split_data(df, target_column: str, test_size: float = 0.1) -> tuple:
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(target_column, axis=1),
        df[target_column],
        test_size=test_size,
        random_state=0,
    )

    return x_train, x_test, y_train, y_test


def create_logistic_regression_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    logreg_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=0, n_jobs=-1),
    )
    logreg_model.fit(x_train, y_train)

    return logreg_model


def create_rnd_forest_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 0,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    rnd_forest_classifier = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(random_state=random_state, n_jobs=n_jobs),
    )
    rnd_forest_classifier.fit(x_train, y_train)

    return rnd_forest_classifier


def get_model_metrics(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float | np.ndarray]:
    test_prediction = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, test_prediction),
        "roc_auc": roc_auc_score(y_test, test_prediction),
        "confusion_matrix": confusion_matrix(y_test, test_prediction),
    }

    return metrics


def save_model(model: Any, path: str, model_name: str) -> None:
    with open(path + "/" + model_name, "wb") as file:
        pickle.dump(model, file)
        print(f"Model {model_name} saved to {path}")


def load_model(path: str, model_name: str):
    with open(path + "/" + model_name, "rb") as file:
        loaded_model = pickle.load(file)

    return loaded_model


def predict_with_model(model: Any, data: pd.DataFrame) -> np.ndarray:
    prediction = model.predict(data)

    return prediction


# learner pipeline:
# get DataFrame
# prepate DataFrame:
#   - encode categorical features
#   - delete unnecessary columns
#   TODO:
#   - if columns contains NaN values, fill them with mean value of column
#   - if columns contains non numeric values throw an error
# split into train and test set
# create models
# save model
"""
target_column = "loan_status"
csv_path = "data"
csv_name = "loan_status.csv"
save_dir = "models"
model_name = "rnd_forest_classifier.pkl"

df_raw = read_dataframe(csv_path, csv_name)
df_encoded = encode_columns(df_raw, encoding_strategy)
df_processed = drop_low_corr_columns(df_encoded, target_column)
x_train, x_test, y_train, y_test = split_data(df_processed)
model = create_rnd_forest_classifier(x_train, y_train)
# optional
# metrics = get_model_metrics(model, x_test, y_test)
save_model(model, save_dir, model_name)
"""
# predictor pipeline:
# get DataFrame
# prepare data for the model
#   - encode categorical features
#   - delete unnecessary columns
#   TODO:
#   - if columns contains NaN values, fill them
#   - if columns contains non numeric values throw an error
# load model
# predict with model
"""
models_dir = "models"

df_raw = read_dataframe(csv_path, csv_name)
df_encoded = encode_columns(df_raw, encoding_strategy)
df_processed = drop_low_corr_columns(df_encoded, target_column)
model = load_model(models_dir, model_name)
prediction = predict_with_model(model, df_processed)
"""


def main():
    print("Hello from ml-service!")


if __name__ == "__main__":
    print(save_model.__code__.co_varnames[: save_model.__code__.co_argcount])
