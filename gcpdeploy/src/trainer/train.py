from google.cloud import storage
import pandas as pd
from dotenv import load_dotenv
import os
import io
import joblib

# import matplotlib.pyplot as plt
# import lime
# import lime.lime_tabular

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_key.json"
client = storage.Client()


def read_file(bucket_name, blob_name):
    """Write and read a blob from GCS using file-like IO"""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with blob.open("r") as f:
        df = f.read()
    final_df = pd.read_csv(io.StringIO(df), sep=",")
    return final_df


def split_time_series_data(df, target_column="close", test_size=0.1, random_state=2024):
    # Sort the DataFrame by date
    df = df.sort_index()

    # Split features and target
    X = df.drop(columns=[target_column, "date"])
    y = df[target_column]

    # Split into train+val and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=2024
    )

    return X_train, X_test, y_train, y_test


def save_and_upload_model(model, local_model_path, gcs_model_path):
    """
    Saves the model locally and uploads it to GCS.

    Parameters:
    model (kmeans): The trained model to be saved and uploaded.
    local_model_path (str): The local path to save the model.
    gcs_model_path (str): The GCS path to upload the model.
    """
    # Save the model locally
    joblib.dump(model, local_model_path)

    storage_client = storage.Client()
    bucket = storage_client.bucket("stock_price_prediction_dataset")
    blob = bucket.blob(gcs_model_path)
    blob.upload_from_filename(local_model_path)
    return True


def train_linear_regression(
    data, target_column="close", test_size=0.1, param_grid=None, tscv=None, random_state=2024
):
    """
    Train a model on the given data and evaluate it using TimeSeriesSplit cross-validation.
    Optionally perform hyperparameter tuning using GridSearchCV.
    Get the best model and its parameters.
    """

    # Split data into train and test sets
    X_train, _, y_train, _ = split_time_series_data(data, target_column, test_size, random_state)

    # Time Series Cross-Validation for train sets (train and validation)
    tscv = TimeSeriesSplit(n_splits=5)

    # Define models and hyperparameters for hyperparameter tuning
    lr_best_model = {}
    model_name = "Ridge Regression"
    model = Ridge()
    param_grid = {"model__alpha": [0.05, 0.1, 0.2, 1.0]}
    print(f"\nTraining {model_name} model...")

    pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("model", model)]  # Handle NaNs in the dataset
    )

    # Perform Grid Search with TimeSeriesSplit if a parameter grid is provided
    if param_grid:
        search = GridSearchCV(
            pipeline, param_grid, cv=tscv, scoring="neg_mean_squared_error", error_score="raise"
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_  # average of valid datasets
        best_params = search.best_params_

    else:
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        best_params = None

    folder = "artifacts"
    if not os.path.exists(folder):
        os.makedirs(folder)

    output_path = f"{folder}/{model_name}.pkl"
    gcs_model_path = f"model_checkpoints/{model_name}.pkl"
    save_and_upload_model(model=best_model, local_model_path=output_path, gcs_model_path=gcs_model_path)
    return best_model, best_params


if __name__ == "__main__":
    bucket_name = "stock_price_prediction_dataset"
    blob_name = "Data/data/Data_pipeline_airflow_dags_data_final_dataset_for_modeling.csv"
    df = read_file(bucket_name, blob_name)
    best_model, best_params = train_linear_regression(
        df, target_column="close", test_size=0.1, random_state=2024
    )

    # print(best_model, best_params)
