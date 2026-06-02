import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def create_logistic_regression_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 0,
    n_jobs: int = -1,
) -> LogisticRegression:
    """
    Create a logistic regression model with a standard scaler preproccessor.

    Args:
        x_train (pd.DataFrame): The training features.
        y_train (pd.Series): The target variable for training.

    Returns:
        LogisticRegression: A trained logistic regression pipeline with scaling.

    """
    logreg_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=random_state, n_jobs=n_jobs),
    )
    logreg_model.fit(x_train, y_train)

    return logreg_model


def create_rnd_forest_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 0,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """
    Create a Random Forest classifier with a standard scaler preproccessor.

    Args:
        x_train (pd.DataFrame): The training features.
        y_train (pd.Series): The target variable for training.
        random_state (int): Controls the randomness of the model initialization. Defaults to 0.
        n_jobs (int): The number of jobs to run in parallel. Defaults to -1, which means using all CPU cores.

    Returns:
        RandomForestClassifier: A trained Random Forest classifier pipeline with scaling.

    Raises:
        ValueError: If x_train or y_train are not valid inputs for the model.
        TypeError: If the input types are incorrect.

    """
    rnd_forest_classifier = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(random_state=random_state, n_jobs=n_jobs),
    )
    rnd_forest_classifier.fit(x_train, y_train)

    return rnd_forest_classifier
