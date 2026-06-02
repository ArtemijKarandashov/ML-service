import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df, target_column: str, test_size: float = 0.1) -> tuple:
    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(target_column, axis=1),
        df[target_column],
        test_size=test_size,
        random_state=0,
    )

    return x_train, x_test, y_train, y_test


def balance_dateframe(
    df: pd.DataFrame,
    target_column: str,
    cutout_threshold: float = 0.65,
    random_state: int = 0,
) -> pd.DataFrame:
    df_zeros = df[df[target_column] == 0]
    df_ones = df[df[target_column] == 1]

    df_minority, df_majoruty = df_zeros, df_ones
    if len(df_ones) <= len(df_zeros):
        df_minority, df_majoruty = df_ones, df_zeros

    delta = len(df_majoruty) - len(df_minority)
    # if skew is extreme (difference is greater than 65%) return original dataframe (data is garbage anyway)
    if delta / len(df) >= cutout_threshold:
        return df

    print(f"zeros:\n{len(df_zeros)}")
    print(f"ones:\n{len(df_ones)}")

    df_majority_downsampled = df_majoruty.sample(
        n=len(df_minority), random_state=random_state
    )
    df_balanced = pd.concat([df_majority_downsampled, df_minority], axis=0)

    return df_balanced


# Unused
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
