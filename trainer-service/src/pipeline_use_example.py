from ml_pipeline.create_models import create_rnd_forest_classifier
from ml_pipeline.dataframe_encoder import encoding_strategy
from ml_pipeline.pipeline import (
    run_prediction_pipeline,
    run_training_pipeline,
)

# --- Demo ---

if __name__ == "__main__":
    pretty_printing = True
    run_training = False
    run_prediction = True

    operational_columns1 = [
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
    ]

    operational_columns2 = [
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
    ]

    with open("ml_pipeline/data/loan_data.csv", "rb") as file:
        csv_bytes = file.read()
    with open("ml_pipeline/models/rnd_forest_classifier.pkl", "rb") as model_file:
        model_bytes = model_file.read()

    config_train = {
        "data": {"csv_bytes": csv_bytes},
        # "model": {"save_dir": "models", "name": "rnd_forest_classifier.pkl"},
        "model": {"model_bytes": model_bytes},
        "target_column": "loan_status",
        "operational_columns": operational_columns1,
        "encoding_strategy": encoding_strategy,
        "create_model": create_rnd_forest_classifier,
        "enable_metrics": True,
    }

    config_prediction = {
        "data": {"csv_bytes": csv_bytes},
        # "model": {"save_dir": "models", "name": "rnd_forest_classifier.pkl"},
        "model": {"model_bytes": model_bytes},
        "operational_columns": operational_columns2,
        "encoding_strategy": encoding_strategy,
    }

    if run_training:
        if pretty_printing:
            print("=" * 50)
            print("ЗАПУСК ОБУЧЕНИЯ")
            print("=" * 50)

        context: dict = {}
        result_train = run_training_pipeline(config_train, context)

        if result_train:
            print("Модель успешно обучена и сохранена.")
            if config_train["enable_metrics"]:
                print(f"model metrics:", result_train["metrics"])
        else:
            print("Обучение завершено с ошибками (см. лог).")

    if run_prediction:
        if pretty_printing:
            print("\n" + "=" * 50)
            print("ЗАПУСК ИНФЕРЕНСА")
            print("=" * 50)

        context: dict = {}
        result_infer = run_prediction_pipeline(config_prediction, context)
        print(f"results: ", result_infer)
