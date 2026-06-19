import json

from ml_pipeline.pipeline_manager import (
    TrainingPipeline,
)

if __name__ == "__main__":
    with open("loan_data.csv", "rb") as file:
        train_csv_bytes = file.read()

    trainer = TrainingPipeline(train_csv_bytes)
    trainer.run_pipeline()

    model_bytes = trainer.get_model_bytes()
    metrics = trainer.get_metrics()

    with open("dist/model.pkl", "wb") as model_file:
        model_file.write(model_bytes)

    with open("dist/metrics.txt", "w") as metrics_file:
        metrics_string = json.dumps(metrics)
        metrics_file.write(metrics_string)
