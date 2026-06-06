import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .create_models import (
    create_logistic_regression_model,
    create_rnd_forest_classifier,
)
from .dataframe_encoder import (
    encode_columns,
    encoding_strategy,
    prediction_allowed_columns,
    training_allowed_columns,
)
from .dataframe_manager import (
    balance_dateframe,
    split_data,
)
from .dataframe_reader import read_csv_from_bytes, read_dataframe
from .logger import get_logger
from .model_manager import (
    get_model_metrics,
    load_model,
    load_model_from_bytes,
    predict_with_model,
    save_model,
    save_model_bytes,
)


class PipelineStep(ABC):
    """Base (and based) step for pipeline."""

    def __init__(self, logger: object = None) -> None:
        """
        Initialize the PipelineStep.

        Args:
            logger (object): The logging object to use. Defaults to None.
                If not provided, a default logger will be used.

        Attributes:
            logger (logging.Logger): The initialized logging object.

        """
        self.logger = logger
        if not logger:
            self.logger = logging.getLogger()

    @abstractmethod
    def execute(self, **kwargs) -> pd.DataFrame | object | None:
        """
        Execute the pipeline step with given parameters.

        Args:
            **kwargs: Additional keyword arguments for execution.

        Returns:
            Union[pd.DataFrame, object, None]: The result of the execution.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        """

    def _handle_error(self, step_name: str, error: Exception) -> None:
        """
        Handle errors that occur during the pipeline step's execution.

        Args:
            step_name (str): The name of the pipeline step where the error occurred.
            error (Exception): The exception object raised during execution.

        """
        msg = f"Step '{step_name}' yielded error: \n{error}."
        self.logger.error(msg)


class LoadDataCSVStep(PipelineStep):
    def __init__(
        self,
        csv_path: str,
        csv_name: str,
    ):
        self.csv_path = csv_path
        self.csv_name = csv_name
        super().__init__()

    def execute(self, context: dict) -> pd.DataFrame | None:
        try:
            self.logger.info(f"Loading data from {self.csv_path}/{self.csv_name}")
            df = read_dataframe(self.csv_path, self.csv_name)

            if df.empty:
                raise ValueError("Loaded pd.DataFrame is empty.")

            context["df_loaded"] = df

        except Exception as e:
            self._handle_error("LoadData", e)
            return None

        else:
            return context


class LoadDataBytesStep(PipelineStep):
    def __init__(self, csv_bytes: bytes):
        self.csv_bytes = csv_bytes
        super().__init__()

    def execute(self, context: dict) -> pd.DataFrame | None:
        try:
            self.logger.info(f"Loading data from bytes")
            df = read_csv_from_bytes(self.csv_bytes)

            if df.empty:
                raise ValueError("Loaded pd.DataFrame is empty.")

            context["df_loaded"] = df

        except Exception as e:
            self._handle_error("LoadData", e)
            return None

        else:
            return context


class EncodeDataStep(PipelineStep):
    def __init__(self, encoding_strategy: dict[str, callable]):
        self.encoding_strategy = encoding_strategy
        super().__init__()

    def execute(self, context: dict) -> pd.DataFrame | None:
        try:
            df_loaded = context["df_loaded"].copy()

            self.logger.info("Statring data encoding...")
            df_encoded = encode_columns(df_loaded, self.encoding_strategy)

            context["df_encoded"] = df_encoded
            del context["df_loaded"]

        except Exception as e:
            self._handle_error("EncodeData", e)
            return None

        else:
            return context


class CleanDataStep(PipelineStep):
    def __init__(self, allowed_columns: list):
        self.allowed_columns = allowed_columns
        super().__init__()

    def execute(self, context: dict) -> pd.DataFrame | None:
        try:
            df_encoded = context["df_encoded"].copy()

            self.logger.info("Deleting columns with low correlation...")
            df_clean = df_encoded.drop(
                df_encoded.columns.difference(self.allowed_columns),
                axis=1,
            )

            context["df_clean"] = df_clean
            del context["df_encoded"]

        except Exception as e:
            self._handle_error("CleanData", e)
            return None

        else:
            return context


class BalanceDataStep(PipelineStep):
    def __init__(self, target_column: str, cutout_threshold: float = 0.7):
        self.target_column = target_column
        self.cutout_threshold = cutout_threshold
        super().__init__()

    def execute(self, context: dict) -> dict:
        try:
            df_clean = context["df_clean"].copy()

            self.logger.info("Balancing dataframe")
            df_clean = balance_dateframe(
                df_clean,
                self.target_column,
                self.cutout_threshold,
            )

            context["df_clean"] = df_clean

        except Exception as e:
            self._handle_error("BalanceData", e)
            return None

        else:
            return context


class SplitDataStep(PipelineStep):
    def __init__(
        self,
        target_column: str = "loan_status",
        test_size: float = 0.1,
        random_state: int = 0,
    ):
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        super().__init__()

    def execute(self, context: dict) -> dict | None:
        try:
            df_clean = context["df_clean"]

            self.logger.info(f"Splitting data (test size: {self.test_size}%).")
            x_train, x_test, y_train, y_test = split_data(
                df_clean,
                target_column=self.target_column,
                test_size=self.test_size,
            )

            context["x_train"] = x_train
            context["x_test"] = x_test
            context["y_train"] = y_train
            context["y_test"] = y_test

            # del context["df_clean"]

        except Exception as e:
            self._handle_error("SplitData", e)
            return None

        else:
            return context


class TrainModelStep(PipelineStep):
    def __init__(self, create_model: callable, random_state: int = 0, n_jobs: int = -1):
        self.create_model = create_model
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__()

    def execute(self, context: dict) -> dict | None:
        try:
            x_train = context["x_train"]
            y_train = context["y_train"]

            self.logger.info("Training model...")
            trained_model = self.create_model(
                x_train,
                y_train,
                self.random_state,
                self.n_jobs,
            )

            context["trained_model"] = trained_model

        except Exception as e:
            self._handle_error("TrainModel", e)
            return None

        else:
            return context


class EvaluateMetricsStep(PipelineStep):
    def __init__(self):
        super().__init__()

    def execute(self, context: dict) -> dict | None:
        try:
            trained_model = context["trained_model"]
            x_test, y_test = context["x_test"], context["y_test"]

            self.logger.info("Evaluate model metrics...")
            metrics = get_model_metrics(trained_model, x_test, y_test)
            # returns dict:
            #   - accuracy_score,
            #   - roc_auc_score,
            #   - confusion_matrix

            context["metrics"] = metrics

        except Exception as e:
            self._handle_error("EvaluateMetrics", e)
            return None

        else:
            return context


class SaveModelStep(PipelineStep):
    def __init__(self, save_dir: str, model_name: str):
        self.save_dir = save_dir
        self.model_name = model_name
        super().__init__()

    def execute(self, context: dict) -> dict | None:
        try:
            trained_model = context["trained_model"]
            self.logger.info(f"Сохранение модели в {self.save_dir}/{self.model_name}")
            save_model(trained_model, self.save_dir, self.model_name)

            context["model_saved"] = True

        except Exception as e:
            self._handle_error("SaveModel", e)
            return None

        else:
            return context


class SaveBytesModel(PipelineStep):
    def __init__(self):
        super().__init__()

    def execute(self, context: dict):
        try:
            model = context["trained_model"]
            self.logger.info("Сохранение модели в байтовый формат")
            model_bytes = save_model_bytes(model)

            context["model_bytes"] = model_bytes

        except Exception as e:
            self._handle_error("SaveBytesModel", e)
            return None

        else:
            return context


class LoadModelStep(PipelineStep):
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        super().__init__()

    def execute(self, context: dict) -> dict | None:
        try:
            self.logger.info(f"Loading model from {self.model_path}/{self}")
            trained_model = load_model(self.model_path, self.model_name)

            context["trained_model"] = trained_model

        except Exception as e:
            self._handle_error("LoadModel", e)
            return None

        else:
            return context


class LoadModelBytesStep(PipelineStep):
    def __init__(self, model_bytes: bytes):
        self.model_bytes = model_bytes
        super().__init__()

    def execute(self, context: dict) -> dict | None:
        try:
            self.logger.info(f"Loading model from bytes")
            trained_model = load_model_from_bytes(self.model_bytes)

            context["trained_model"] = trained_model

        except Exception as e:
            self._handle_error("LoadModelBytes", e)
            return None

        else:
            return context


class PredictionStep(PipelineStep):
    def __init__(self):
        super().__init__()

    def execute(self, context: dict) -> dict | None:
        try:
            trained_model = context["trained_model"]
            df_clean = context["df_clean"].copy()

            prediction = predict_with_model(trained_model, df_clean)

            context["prediction"] = prediction

        except Exception as e:
            self._handle_error("Predict", e)
            return None

        else:
            return context


class StrategyPipeline:
    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps

    def run(self, context) -> dict:
        for step in self.steps:
            context = step.execute(context)
            # print("Context: ", context)
            # try:
            #     print("Context: ", context["df_clean"].info())
            # except Exception:
            #     pass
        return context


def build_training_pipeline(
    csv_bytes: bytes,  # resieved from controller
    enable_metrics: bool = True,
    allowed_columns: list = training_allowed_columns,
    encoding_strategy: dict = encoding_strategy,
    target_column: str = "loan_status",
    create_model: callable = create_rnd_forest_classifier,
) -> dict("model_bytes"):
    """Fabric function to build a training pipeline."""
    # learner pipeline:
    # get DataFrame
    # prepate DataFrame:
    #   - encode categorical features
    #   - delete unnecessary columns
    #   TODO:
    #   - if columns contains NaN values, fill them with mean value of column
    #   - if columns contains non numeric values throw an error
    # split into train and test set
    # create model
    # train model
    # optional:
    #   - evaluate metrics on train and test set
    # save model

    pipeline = StrategyPipeline(
        [
            LoadDataBytesStep(csv_bytes=csv_bytes),
            EncodeDataStep(encoding_strategy=encoding_strategy),
            CleanDataStep(allowed_columns=allowed_columns),
            BalanceDataStep(target_column=target_column),
            SplitDataStep(target_column=target_column, test_size=0.1),
            TrainModelStep(create_model=create_model),
            EvaluateMetricsStep() if enable_metrics else None,
            # SaveModelStep(save_dir=save_dir, model_name=model_name),
            SaveBytesModel(),
        ],
    )

    return pipeline


def run_training_pipeline(
    config: dict("csv_bytes"),
    logger: object,
    context: dict,
) -> dict("model_bytes", "metrics"):
    """Wrapper for config based pipeline."""
    if logger is None:
        logger = get_logger()
    logger.info("Starting model training pipeline...")

    # if context is not dict or not defined
    if isinstance(context, dict):
        context = {}

    try:
        training_pipeline = build_training_pipeline(
            csv_bytes=config["csv_bytes"],
            # model_name=config["model"]["name"],
            # save_dir=config["model"]["save_dir"],
            # encoding_strategy=config["encoding_strategy"],
            # allowed_columns=config["allowed_columns"],
            # target_column=config["target_column"],
            # enable_metrics=config["enable_metrics"],
            # create_model=config["create_model"],
        )

        logger.info("Executing pipeline...")

        result = training_pipeline.run(context)

        if result["model_saved"]:
            logger.info("Pipeline seccessfully executed")

    except Exception as e:
        logger.exception(f"Error during pipeline execution. Error: {e}")
        return None

    else:
        return result


def build_prediction_pipeline(
    csv_bytes: bytes,  # resieved from controller
    model_bytes: object,  # resieved from controller
    encoding_strategy: dict = encoding_strategy,
    allowed_columns: list = prediction_allowed_columns,
):
    """Fabric function to build a prediction pipeline."""
    # predictor pipeline:
    # get pd.DataFrame
    # prepare data for the model
    #   - encode categorical features
    #   - delete unnecessary columns
    #   TODO:
    #   - if columns contains NaN values, fill them
    #   - if columns contains non numeric values throw an error
    # load model
    # predict with model

    pipeline = StrategyPipeline(
        [
            LoadDataBytesStep(csv_bytes=csv_bytes),
            EncodeDataStep(encoding_strategy=encoding_strategy),
            CleanDataStep(allowed_columns=allowed_columns),
            LoadModelBytesStep(model_bytes=model_bytes),
            PredictionStep(),
        ],
    )

    return pipeline


def run_prediction_pipeline(
    config: dict("csv_bytes", "model_bytes"),
    context: dict,
) -> np.array:
    """Запуск процесса предсказания."""
    logger = get_logger()
    logger.info("Starting run_prediction_pipeline...")

    try:
        prediction_pipeline = build_prediction_pipeline(
            csv_bytes=config["csv_bytes"],
            model_bytes=config["model_bytes"],
            # allowed_columns=config["allowed_columns"],
            # encoding_strategy=config["encoding_strategy"],
        )

        prediction = prediction_pipeline.run(context)

    except Exception as e:
        logger.exception(f"Error in run_prediction_pipeline: {e}")
        return None

    else:
        logger.info("prediction pipeline completed successfully.")
        return prediction["prediction"]
