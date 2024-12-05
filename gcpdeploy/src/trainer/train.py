from google.cloud import storage
import pandas as pd
from dotenv import load_dotenv
import os
import io
import joblib
import wandb
import yaml
import matplotlib.pyplot as plt
import logging

from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.utils import resample

from datetime import datetime

with open("config.yaml") as f:
    config = yaml.safe_load(f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_key.json"
client = storage.Client()


try:
    key = config["WANDB_API_KEY"]
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb.login(key=key)
except Exception as e:
    print(f"--- was trying to log in Weights and Biases... e={e}")


def read_file(bucket_name, blob_name):
    """Write and read a blob from GCS using file-like IO"""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with blob.open("r") as f:
        df = f.read()
    final_df = pd.read_csv(io.StringIO(df), sep=",")
    return final_df


def detect_bias(data_train):
    X_train = data_train.drop(columns=["date", "close"])
    y_train = data_train["close"]

    # split X_train to train and validation
    X_train = X_train.iloc[: -int(len(X_train) * 0.1)]
    y_train = y_train.iloc[: -int(len(y_train) * 0.1)]
    X_val = X_train.iloc[-int(len(X_train) * 0.1) :]
    y_val = y_train.iloc[-int(len(y_train) * 0.1) :]

    # Define features and target
    features = [col for col in data_train.columns if col != "close" and col != "date"]
    target = "close"

    # Add the target column to the training set for resampling
    train_data = X_train.copy()
    train_data["close"] = y_train
    # Add the target to the training features for resampling

    # Define the slice condition based on VIX values
    vix_median = pd.to_numeric(train_data["VIXCLS"], errors="coerce").median()
    high_vix_data = train_data[train_data["VIXCLS"] > vix_median]
    low_vix_data = train_data[train_data["VIXCLS"] <= vix_median]

    # Upsample high-error slice data
    high_vix_upsampled = resample(high_vix_data, replace=True, n_samples=len(low_vix_data), random_state=42)
    balanced_train_data = pd.concat([high_vix_upsampled, low_vix_data])

    # Separate features and target after balancing
    X_train_balanced = balanced_train_data.drop(columns=["close"])  # Drop the target from balanced data
    y_train_balanced = balanced_train_data["close"]  # Get the target values

    # Train the ElasticNet model on the balanced data
    model = Ridge(alpha=0.05)
    model.fit(X_train_balanced, y_train_balanced)

    # Predict and evaluate on the test set
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    logging.info(f"Linear Regression Test MSE with Resampling: {mse}")
    logging.info(f"Linear Regression Test MAE with Resampling: {mae}")

    local_file_path = "artifacts/bias_detection.txt"
    # save mse and mae to a file
    with open(local_file_path, "w") as f:
        f.write(f"Linear Regression Test MSE with Resampling: {mse}")
        f.write(f"Linear Regression Test MAE with Resampling: {mae}")

    # upload to gcs
    storage_client = storage.Client()
    bucket = storage_client.bucket("stock_price_prediction_dataset")
    blob = bucket.blob("artifacts/bias_detection_metrics.txt")
    blob.upload_from_filename(local_file_path)
    return True


def feature_importance_analysis(model, X_train, y_train):
    # Permutation feature importance
    perm_importance = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)

    # Sort features by importance
    sorted_idx = perm_importance.importances_mean.argsort()

    # Select top 10 features
    top_10_idx = sorted_idx[-10:]

    plt.figure(figsize=(10, 6))
    plt.barh(range(10), perm_importance.importances_mean[top_10_idx])
    plt.yticks(range(10), X_train.columns[top_10_idx])
    plt.title("Top 10 Features - Permutation Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    local_pic_path = "artifacts/feature_importance.png"
    plt.savefig(local_pic_path)

    # upload to gcs
    storage_client = storage.Client()
    bucket = storage_client.bucket("stock_price_prediction_dataset")
    blob = bucket.blob("artifacts/feature_importance.png")
    blob.upload_from_filename(local_pic_path)
    return True


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


def train_linear_regression(data_train, target_column="close", param_grid=None, tscv=None):
    """
    Train a model on the given data and evaluate it using TimeSeriesSplit cross-validation.
    Optionally perform hyperparameter tuning using GridSearchCV.
    Get the best model and its parameters.
    """

    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = wandb.init(project="stock-price-prediction", name=f"experiment_{cur_time}")

    # drop nan values
    data_train = data_train.dropna()
    X_train = data_train.drop(columns=["date", target_column])
    y_train = data_train["close"]

    # Time Series Cross-Validation for train sets (train and validation)
    tscv = TimeSeriesSplit(n_splits=5)

    # Define models and hyperparameters for hyperparameter tuning
    model_name = "model"
    model = Ridge()
    param_grid = {"model__alpha": [0.05, 0.1, 0.2, 1.0]}
    print(f"\nTraining {model_name} model...")

    pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("model", model)]  # Handle NaNs in the dataset
    )

    # Perform Grid Search with TimeSeriesSplit if a parameter grid is provided
    if param_grid:
        search = GridSearchCV(
            pipeline, param_grid, cv=tscv, scoring="neg_mean_squared_error", error_score="raise", verbose=3
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_  # average of valid datasets
        best_params = search.best_params_

        # log wandb
        log_metrics = {}
        log_metrics["params"] = search.cv_results_["params"]
        log_metrics["split0_valid_score"] = search.cv_results_["split0_test_score"]
        log_metrics["split1_valid_score"] = search.cv_results_["split1_test_score"]
        log_metrics["split2_valid_score"] = search.cv_results_["split2_test_score"]
        log_metrics["split3_valid_score"] = search.cv_results_["split3_test_score"]
        log_metrics["split4_valid_score"] = search.cv_results_["split4_test_score"]
        log_metrics["mean_valid_score"] = search.cv_results_["mean_test_score"]
        log_metrics["std_valid_score"] = search.cv_results_["std_test_score"]
        log_metrics["rank_valid_score"] = search.cv_results_["rank_test_score"]
        log_metrics = {}

        timesteps = [0.05, 0.1, 0.2, 1.0]
        values = search.cv_results_["mean_test_score"]
        for i in range(len(timesteps)):
            wandb.log({f"apha_{timesteps[i]}": values[i]})

        for key, value in log_metrics.items():
            for i in range(len(value)):
                wandb.log({key: value[i]})

        wandb.finish()
        print(f"log_metrics: {log_metrics}")

    else:
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        best_params = None

    folder = "artifacts"
    if not os.path.exists(folder):
        os.makedirs(folder)

    feature_importance_analysis(best_model, X_train, y_train)
    detect_bias(data_train)

    output_path = f"{folder}/{model_name}.joblib"
    gcs_model_path = f"model_checkpoints/{model_name}.joblib"
    save_and_upload_model(model=best_model, local_model_path=output_path, gcs_model_path=gcs_model_path)
    return best_model, best_params

    return best_model, best_params


if __name__ == "__main__":
    print("starting....")
    bucket_name = "stock_price_prediction_dataset"
    blob_name = "Data/pipeline/airflow/dags/data/scaled_data_train.csv"
    df = read_file(bucket_name, blob_name)
    best_model, best_params = train_linear_regression(df, target_column="close")
    print("Done!!!")
