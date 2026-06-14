# import logging
import numpy as np

from .create_models import create_rnd_forest_classifier
from .dataframe_encoder import (
    encoding_strategy,
    prediction_allowed_columns,
    training_allowed_columns,
)
from .logger import get_logger
from .pipeline_steps import (
    BalanceDataStep,
    CleanDataStep,
    EncodeDataStep,
    EvaluateMetricsStep,
    LoadDataBytesStep,
    LoadModelBytesStep,
    PredictionStep,
    SaveBytesModel,
    SplitDataStep,
    StrategyPipeline,
    TrainModelStep,
)


class TrainingPipeline:
    def __init__(
        self,
        csv_bytes: bytes,
        config: dict | None = None,
    ):
        self.result = None
        self.csv_bytes = csv_bytes

        self.set_configuration(config)
        self.pipeline = self.build_pipeline()

    def build_pipeline(self) -> StrategyPipeline:
        """Fabric function to build a training pipeline."""
        pipeline = StrategyPipeline(
            [
                LoadDataBytesStep(csv_bytes=self.csv_bytes),
                EncodeDataStep(encoding_strategy=self.encoding_strategy),
                CleanDataStep(allowed_columns=self.allowed_columns),
                BalanceDataStep(target_column=self.target_column),
                SplitDataStep(
                    target_column=self.target_column,
                    test_size=self.test_size,
                ),
                TrainModelStep(model_func=self.model_func),
                EvaluateMetricsStep() if self.enable_metrics else None,
                SaveBytesModel(),
            ],
        )

        return pipeline

    def run_pipeline(
        self,
        logger: object | None = None,  # provide logger object for logging purposes
        context: dict | None = None,  # provide context for specisic pipeline behavior
    ) -> dict("model_bytes", "metrics" | None):
        if logger is None:
            logger = get_logger()

        # if context is not dict or not defined
        if not isinstance(context, dict):
            context = {}

        logger.info("Executing pipeline...")
        result = self.pipeline.run(context)

        self.model_bytes: bytes = result.get("model_bytes")
        if self.enable_metrics:
            self.metrics: dict = result.get("metrics")

        return result

    def get_model_bytes(self) -> bytes:
        """Get the trained model in bytes."""
        if not hasattr(self, "model_bytes"):
            raise RuntimeError(
                "Cannot get model bytes before running pipeline. Call run_pipeline() first.",
            )
        return self.model_bytes

    def get_metrics(self) -> dict | None:
        """Get evaluation metrics."""
        if not hasattr(self, "metrics"):
            raise RuntimeError(
                "Cannot get metrics before running pipeline. Call run_pipeline() first.",
            )
        return self.metrics

    def set_configuration(self, config: dict | None):
        """Set configuration for the pipeline."""
        if not config:
            config = {}

        self.encoding_strategy: dict = config.get(
            "encoding_strategy",
            encoding_strategy,
        )

        self.allowed_columns: list = config.get(
            "allowed_columns",
            training_allowed_columns,
        )

        self.target_column: str = config.get(
            "target_column",
            "loan_status",
        )

        self.test_size: float = config.get(
            "test_size",
            0.1,
        )

        self.model_func: callable = config.get(
            "model_func",
            create_rnd_forest_classifier,
        )

        self.enable_metrics: bool = config.get(
            "enable_metrics",
            True,
        )


class PredictionPipeline:
    def __init__(
        self,
        csv_bytes: bytes,
        model_bytes: bytes,
        config: dict | None = None,
    ):
        self.csv_bytes = csv_bytes
        self.model_bytes = model_bytes

        self.set_configuration(config)
        self.pipeline = self.build_pipeline()

    def build_pipeline(self) -> StrategyPipeline:
        pipeline = StrategyPipeline(
            [
                LoadDataBytesStep(csv_bytes=self.csv_bytes),
                EncodeDataStep(encoding_strategy=self.encoding_strategy),
                CleanDataStep(allowed_columns=self.allowed_columns),
                LoadModelBytesStep(model_bytes=self.model_bytes),
                PredictionStep(),
            ],
        )

        return pipeline

    def run_pipeline(
        self,
        logger: object | None = None,  # provide logger object for logging purposes
        context: dict | None = None,  # provide context for specisic pipeline behavior
    ) -> np.array:
        if logger is None:
            logger = get_logger()

        # if context is not dict or not defined
        if not isinstance(context, dict):
            context = {}

        logger.info("Executing pipeline...")
        result = self.pipeline.run(context)

        self.prediction: np.array = result.get("prediction")

        return result

    def get_prediction(self) -> np.array:
        """Get the prediction result."""
        if not hasattr(self, "prediction"):
            raise RuntimeError(
                "Cannot get prediction before running the pipeline. Call run_pipeline() first.",
            )
        return self.prediction

    def set_configuration(self, config: dict | None = None):
        """Set configuration for the pipeline."""
        if not config:
            config = {}

        self.encoding_strategy: dict = config.get(
            "encoding_strategy",
            encoding_strategy,
        )
        self.allowed_columns: list = config.get(
            "allowed_columns",
            prediction_allowed_columns,
        )
