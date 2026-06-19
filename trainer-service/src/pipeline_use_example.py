from ml_pipeline.pipeline_manager import (
    PredictionPipeline,
    TrainingPipeline,
)

# --- Demo ---

if __name__ == "__main__":
    run_training = False
    run_prediction = False

    with open("ml_pipeline/data/prediction_data_copy.csv", "rb") as file:
        pred_csv_bytes = file.read()

    with open("ml_pipeline/data/loan_data.csv", "rb") as file:
        train_csv_bytes = file.read()

    with open("ml_pipeline/models/rnd_forest_classifier.pkl", "rb") as model_file:
        model_bytes = model_file.read()

    # ====== TRAINING PIPELINE =====
    if run_training:
        print("=" * 20)
        print("ЗАПУСК ОБУЧЕНИЯ")
        print("=" * 20)

        trainer = TrainingPipeline(train_csv_bytes)
        trainer.run_pipeline()
        # training_result = trainer.run_pipeline()

        # ====== GET MODEL AND METRICS =====
        print("Модель успешно получена.")
        model_bytes = trainer.get_model_bytes()
        metrics = trainer.get_metrics()

        for metric in metrics.items():
            print(metric)

    # ====== INFERENCE PIPELINE ======
    if run_prediction:
        print("\n" + "=" * 20)
        print("ЗАПУСК ИНФЕРЕНСА")
        print("=" * 20)

        predictor = PredictionPipeline(pred_csv_bytes, model_bytes)
        predictor.run_pipeline()

        # ====== GET PREDICTION =====
        prediction = predictor.get_prediction()

        print("Предсказание успешно получено.")
        print(prediction)
