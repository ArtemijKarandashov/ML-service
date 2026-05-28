import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from ml_utilities import (
    create_rnd_forest_classifier,
    encode_columns,
    encoding_strategy,
    get_model_metrics,
    load_model,
    predict_with_model,
    read_dataframe,
    save_model,
    split_data,
)


def setup_logger(
    file_path: str = "logs",
    file_name: str = "ml_pipeline.log",
    level: int = logging.INFO,
):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    filename = os.path.join(file_path, file_name)

    if filename and not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    if filename is None:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Консольный вывод (надо?)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class PipelineStep(ABC):
    """Base (and based) step for pipeline."""

    @abstractmethod
    def execute(self, **kwargs) -> Optional[Union[pd.DataFrame, object]]:
        pass

    def _handle_error(self, step_name: str, error: Exception):
        logging.error(f"Step '{step_name}' yeld error: \n{error}.")


class LoadDataStep(PipelineStep):
    def __init__(self, csv_path: str, csv_name: str):
        self.csv_path = csv_path
        self.csv_name = csv_name

    def execute(self, context: dict) -> Optional[pd.DataFrame]:
        try:
            logging.info(f"Loading data from {self.csv_path}/{self.csv_name}")
            df = read_dataframe(self.csv_path, self.csv_name)

            if df.empty:
                raise ValueError("Loaded pd.DataFrame is empty.")

            context["df_loaded"] = df

        except Exception as e:
            self._handle_error("LoadData", e)
            return None

        else:
            return context


class EncodeDataStep(PipelineStep):
    def __init__(self, encoding_strategy: Dict[str, Callable]):
        self.encoding_strategy = encoding_strategy

    def execute(self, context: dict) -> Optional[pd.DataFrame]:
        try:
            df_loaded = context["df_loaded"].copy()

            logging.info("Statring data encoding...")
            df_encoded = encode_columns(df_loaded, self.encoding_strategy)

            context["df_encoded"] = df_encoded
            del context["df_loaded"]

        except Exception as e:
            self._handle_error("EncodeData", e)
            return None

        else:
            return context


class CleanDataStep(PipelineStep):
    def __init__(self, operational_columns: list):
        self.operational_columns = operational_columns

    def execute(self, context: dict) -> Optional[pd.DataFrame]:
        try:
            df_encoded = context["df_encoded"].copy()

            logging.info(f"Deleting columns with low correlation...")
            df_clean = df_encoded.drop(
                df_encoded.columns.difference(self.operational_columns),
                axis=1,
            )

            context["df_clean"] = df_clean
            del context["df_encoded"]

        except Exception as e:
            self._handle_error("CleanData", e)
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

    def execute(
        self, context: dict
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        try:
            df_clean = context["df_clean"]

            logging.info(f"Splitting data (test size: {self.test_size}%).")
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

    def execute(self, context: dict) -> Optional[object]:
        try:
            # create_model = context["create_model"]
            x_train = context["x_train"]
            y_train = context["y_train"]

            logging.info("Training model...")
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
        pass

    def execute(self, context: dict) -> Optional[Dict]:
        try:
            trained_model = context["trained_model"]
            x_test, y_test = context["x_train"], context["y_train"]

            logging.info("Evaluate model metrics...")
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

    def execute(self, context: dict) -> Optional[Dict]:
        try:
            trained_model = context["trained_model"]
            logging.info(f"Сохранение модели в {self.save_dir}/{self.model_name}")
            save_model(trained_model, self.save_dir, self.model_name)

            context["model_saved"] = True

        except Exception as e:
            self._handle_error("SaveModel", e)
            return None

        else:
            return context


class LoadModelStep(PipelineStep):
    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name

    def execute(self, context: dict):
        try:
            logging.info(f"Loading model from {self.model_path}/{self}")
            trained_model = load_model(self.model_path, self.model_name)

            context["trained_model"] = trained_model

        except Exception as e:
            self._handle_error("LoadModel", e)
            return None

        else:
            return context


class PredictionStep(PipelineStep):
    def __init__(self):
        pass

    def execute(self, context: dict):
        try:
            trained_model = context["trained_model"]
            df_clean = context["df_clean"].copy()

            prediction = predict_with_model(trained_model, df_clean)

            context["prediction"] = prediction
            context["status"] = True

        except Exception as e:
            self._handle_error("Predict", e)
            return None

        else:
            return context


class StrategyPipeline:
    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps

    def run(self, context):
        for step in self.steps:
            context = step.execute(context)
        return context


def build_training_pipeline(
    csv_path: str,
    csv_name: str,
    model_name: str,
    save_dir: str,
    encoding_strategy: dict,
    operational_columns: list,
    target_column: str,
    enable_metrics: bool = True,
    create_model: callable = create_rnd_forest_classifier,
):
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
            LoadDataStep(csv_path=csv_path, csv_name=csv_name),
            EncodeDataStep(encoding_strategy=encoding_strategy),
            CleanDataStep(operational_columns=operational_columns),
            SplitDataStep(target_column=target_column, test_size=0.1),
            TrainModelStep(create_model=create_model),
            # May break pipeline
            # EvaluateMetricsStep() if enable_metrics else None,
            SaveModelStep(save_dir=save_dir, model_name=model_name),
        ],
    )

    return pipeline


def run_training_pipeline(config: dict, context: dict):
    """Wrapper for config based pipeline."""
    logger = setup_logger("ml_pipeline.log")
    logger.info("Starting model training pipeline...")

    try:
        training_pipeline = build_training_pipeline(
            csv_path=config["data"]["csv_path"],
            csv_name=config["data"]["csv_name"],
            model_name=config["model"]["name"],
            save_dir=config["model"]["save_dir"],
            encoding_strategy=config["encoding_strategy"],
            operational_columns=config["operational_columns"],
            target_column=config["target_column"],
            enable_metrics=config["enable_metrics"],
            create_model=config["create_model"],
        )

        logger.info("Executing pipeline...")

        result = training_pipeline.run(context)

        if result["model_saved"]:
            logger.info("Pipeline seccessfully executed")

        return result

    except Exception as e:
        logger.exception(f"Crite error during pipeline execution. Error: {e}")
        return None


def build_inference_pipeline(
    csv_path: str,
    csv_name: str,
    model_path: str,
    model_name: str,
    operational_columns: list,
    encoding_strategy: dict,
    # x_train_shape_descriptor=None,  # Data structure dicriptor for validation
):
    """Fabric function to build a inference pipeline."""
    # pipeline = StrategyPipeline(target_column=target_column)
    # pipeline.add_step(LoadDataStep(csv_path="data", csv_name="loan_status.csv"))
    # pipeline.add_step(EncodeDataStep(encoding_strategy=encoding_stategy))
    # pipeline.add_step(CleanDataStep(target_column=target_column))

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
            LoadDataStep(csv_path=csv_path, csv_name=csv_name),
            EncodeDataStep(encoding_strategy=encoding_strategy),
            CleanDataStep(operational_columns=operational_columns),
            LoadModelStep(model_path=model_path, model_name=model_name),
            PredictionStep(),
        ],
    )

    return pipeline


def run_inference_pipeline(config: dict, context: dict):
    """Запуск процесса предсказания."""
    logger = setup_logger("ml_inference.log")
    logger.info("Starting run_inference_pipeline...")

    try:
        inference_pipeline = build_inference_pipeline(
            csv_path=config["data"]["csv_path"],
            csv_name=config["data"]["csv_name"],
            model_path=config["model"]["save_dir"],
            model_name=config["model"]["name"],
            operational_columns=config["operational_columns"],
            encoding_strategy=config["encoding_strategy"],
        )

        prediction = inference_pipeline.run(context)

    except Exception as e:
        logger.exception(f"Error in run_inference_pipeline: {e}")
        return None

    else:
        logger.info("Inference pipeline completed successfully.")
        return {"status": prediction["status"], "prediction": prediction["prediction"]}


# --- Demo ---

if __name__ == "__main__":
    pretty_printing = True
    run_training = False
    run_inference = True

    config_train = {
        # training config UNFINISHED
        "data": {"csv_path": "data", "csv_name": "loan_data.csv"},
        "model": {"save_dir": "models", "name": "rnd_forest_classifier.pkl"},
        "target_column": "loan_status",
        "operational_columns": [
            "loan_status",
            "person_age",
            "person_income",
            "person_emp_exp",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            "credit_score",
            "person_gender_encoded",
            "Associate",
            "Bachelor",
            "Master",
            "MORTGAGE",
            "OTHER",
            "OWN",
            "RENT",
            "DEBTCONSOLIDATION",
            "EDUCATION",
            "HOMEIMPROVEMENT",
            "MEDICAL",
            "PERSONAL",
            "VENTURE",
            "previous_loan_defaults_on_file_encoded",
        ],
        "encoding_strategy": encoding_strategy,
        "create_model": create_rnd_forest_classifier,
        "enable_metrics": True,
    }

    config_inference = {
        # inference config UNFINISHED
        "data": {"csv_path": "data", "csv_name": "loan_data_copy.csv"},
        "model": {"save_dir": "models", "name": "rnd_forest_classifier.pkl"},
        "operational_columns": [
            "person_age",
            "person_income",
            "person_emp_exp",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            "credit_score",
            "person_gender_encoded",
            "Associate",
            "Bachelor",
            "Master",
            "MORTGAGE",
            "OTHER",
            "OWN",
            "RENT",
            "DEBTCONSOLIDATION",
            "EDUCATION",
            "HOMEIMPROVEMENT",
            "MEDICAL",
            "PERSONAL",
            "VENTURE",
            "previous_loan_defaults_on_file_encoded",
        ],
        "encoding_strategy": encoding_strategy,
    }

    # try:
    if run_training:
        if pretty_printing:
            print("=" * 50)
            print("ЗАПУСК ОБУЧЕНИЯ")
            print("=" * 50)

        context: dict = {}
        result_train = run_training_pipeline(config_train, context)
        if result_train:
            print("Модель успешно обучена и сохранена.")
        else:
            print("Обучение завершено с ошибками (см. лог).")
    # except Exception as e:
    #     print(f"Исключение во время обучения: {e}")

    if run_inference:
        if pretty_printing:
            print("\n" + "=" * 50)
            print("2. ЗАПУСК ИНФЕРЕНСА")
            print("=" * 50)

        context: dict = {}
        result_infer = run_inference_pipeline(config_inference, context)
        print(f"results: ", result_infer)
