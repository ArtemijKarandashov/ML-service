import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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
