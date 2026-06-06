from ml_pipeline.create_models import create_rnd_forest_classifier
from ml_pipeline.dataframe_encoder import encoding_strategy
from ml_pipeline.pipeline import (
    run_prediction_pipeline,
    run_training_pipeline,
)

# --- Demo ---

if __name__ == "__main__":
    pretty_printing = True
    run_training = True
    run_prediction = True

    with open("ml_pipeline/data/loan_data.csv", "rb") as file:
        csv_bytes = file.read()
    with open("ml_pipeline/models/rnd_forest_classifier.pkl", "rb") as model_file:
        model_bytes = model_file.read()

    config_train = {
        "csv_bytes": csv_bytes,
    }

    config_prediction = {
        "csv_bytes": csv_bytes,
        "model_bytes": model_bytes,
    }

    if run_training:
        if pretty_printing:
            print("=" * 50)
            print("ЗАПУСК ОБУЧЕНИЯ")
            print("=" * 50)

        result_train = run_training_pipeline(config_train)

        if result_train:
            print("Модель успешно обучена и сохранена.")
            if config_train["enable_metrics"]:
                print("model metrics:", result_train["metrics"])
        else:
            print("Обучение завершено с ошибками (см. лог).")

    if run_prediction:
        if pretty_printing:
            print("\n" + "=" * 50)
            print("ЗАПУСК ИНФЕРЕНСА")
            print("=" * 50)

        result_infer = run_prediction_pipeline(config_prediction)
        print("results: ", result_infer)
