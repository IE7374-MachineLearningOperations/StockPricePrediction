from google.cloud import storage
import pandas as pd
import numpy as np
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

with open('config.yaml') as f:
    config = yaml.safe_load(f)



os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_key.json"
client = storage.Client()



try:
    # wandb.require("legacy-service")
    # os.environ["WANDB__REQUIRE_LEGACY_SERVICE"]="TRUE"
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

def model_bias_detection(data_train, model_name, model, tolerance=0.1, threshold=0.9):
    """
    Perform model bias detection, including slicing, metrics evaluation, and evidence validation.
    """
    def define_slices(data_train):
        high_volume = data_train[data_train['volume'] > data_train['volume'].quantile(0.75)]
        low_volume = data_train[data_train['volume'] <= data_train['volume'].quantile(0.25)]
        overbought = data_train[data_train['RSI'] > 70]
        oversold = data_train[data_train['RSI'] < 30]
        bullish_market = data_train[data_train['SP500'] > data_train['SP500'].mean()]
        high_volatility = data_train[data_train['VIXCLS'] > data_train['VIXCLS'].mean()]
        return {
            'High Volume': high_volume,
            'Low Volume': low_volume,
            'Overbought': overbought,
            'Oversold': oversold,
            'Bullish Market': bullish_market,
            'High Volatility': high_volatility
        }

    def evaluate_slice(slice_data, model, X_columns, y_column):
        if len(slice_data) < 10:  # Skip slices with insufficient data
            return {'R²': None}
        X_slice = slice_data[X_columns]
        y_slice = slice_data[y_column]
        r2 = r2_score(y_slice, model.predict(X_slice))
        return {'R²': r2}

    def validate_prediction_accuracy(slices, model, X_columns, y_column, tolerance, threshold):
        results = {}
        for slice_name, slice_data in slices.items():
            if len(slice_data) < 10:  # Skip slices with insufficient data
                results[slice_name] = {'Pass': None, 'Accuracy': None}
                continue

            X_slice = slice_data[X_columns]
            y_slice = slice_data[y_column]
            y_pred_slice = model.predict(X_slice)
            errors = np.abs(y_pred_slice - y_slice) / np.abs(y_slice)
            accurate_predictions = (errors <= tolerance).mean()
            results[slice_name] = {
                'Accuracy': accurate_predictions,
                'Pass': accurate_predictions >= threshold
            }

        return results

    slices = define_slices(data_train)
    features = [col for col in data_train.columns if col not in ['close', 'date']]
    target = 'close'
    metrics_by_slice = {
        slice_name: evaluate_slice(slice_data, model, features, target)
        for slice_name, slice_data in slices.items()
    }

    accuracy_results = validate_prediction_accuracy(slices, model, features, target, tolerance, threshold)
    consistent_r2 = all(
        slice_metrics['R²'] is None or slice_metrics['R²'] >= 0.90
        for slice_metrics in metrics_by_slice.values()
    )
    no_missing_data_bias = all(
        any(value is not None for value in slice_metrics.values())
        or len(slices[slice_name]) == 0
        for slice_name, slice_metrics in metrics_by_slice.items()
    )
    all_slices_passed_accuracy = all(
        result['Pass'] for result in accuracy_results.values() if result['Pass'] is not None
    )

    # Prepare results for logging and saving
    bias_results = f"\nValidating Supporting Evidence for No Bias ({model_name}):\n"
    bias_results += f"Consistent R² Across Slices: {'Passed' if consistent_r2 else 'Failed'}\n"
    bias_results += f"No Missing Data Bias: {'Passed' if no_missing_data_bias else 'Failed'}\n\n"
    bias_results += "Slice-Level Prediction Accuracy:\n"
    for slice_name, result in accuracy_results.items():
        if result['Pass'] is not None:
            bias_results += f"{slice_name}: Accuracy = {result['Accuracy']:.2%}, Pass = {result['Pass']}\n"
        else:
            bias_results += f"{slice_name}: Insufficient Data\n"

    if consistent_r2 and no_missing_data_bias and all_slices_passed_accuracy:
        bias_results += f"\nConclusion: The model '{model_name}' shows no significant bias across all slices."
    else:
        bias_results += f"\nConclusion: The model '{model_name}' shows potential bias in certain slices."

    print("Model bias detection completed!")
    logging.info(bias_results)

    # Save results locally
    local_file_path = "artifacts/bias_detection_results.txt"
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
    with open(local_file_path, "w") as f:
        f.write(bias_results)

    storage_client = storage.Client()
    bucket = storage_client.bucket("stock_price_prediction_dataset")
    blob = bucket.blob("artifacts/bias_detection_results.txt")
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
    
    
    #upload to gcs
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
    best_model, best_params = train_linear_regression(
        df, target_column="close"
    )
    model_bias_detection(data_train=df, model_name="Ridge Regression", model=best_model)
    print("Done!!!")