import numpy as np
import pandas as pd
import sys
import os
import logging
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from google.cloud import storage


sys.path.append(os.path.abspath("pipeline/airflow"))
sys.path.append(os.path.abspath("."))

from dags.src.download_data import (
    get_yfinance_data,
    get_fama_french_data,
    get_ads_index_data,
    get_sp500_data,
    get_fred_data,
    merge_data,
)
from dags.src.convert_column_dtype import convert_type_of_columns
from dags.src.keep_latest_data import keep_latest_data
from dags.src.remove_weekend_data import remove_weekends
from dags.src.handle_missing import fill_missing_values
from dags.src.correlation import removing_correlated_variables
from dags.src.lagged_features import add_lagged_features
from dags.src.feature_interactions import add_feature_interactions
from dags.src.technical_indicators import add_technical_indicators
from dags.src.scaler import scaler
from dags.src.upload_blob import upload_blob


# Initialize variables
abs_path = os.path.abspath(__file__)
dir = os.path.dirname(abs_path)
dir = os.path.dirname(dir)
dir = os.path.dirname(dir)
path = os.path.join(dir, "service_key_gcs.json")

if os.path.exists(path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
    storage_client = storage.Client()
else:
    storage_client = None
    logging.warning("------- Service key not found!")


# Define the data preparation function
def prepare_data(data, test_size=0.2):
    X = data.drop(["close", "date"], axis=1)
    y = data["close"]

    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


# def save_and_upload_model(model, local_model_path, gcs_model_path):
#     """
#     Saves the model locally and uploads it to GCS.

#     Parameters:
#     model (kmeans): The trained model to be saved and uploaded.
#     local_model_path (str): The local path to save the model.
#     gcs_model_path (str): The GCS path to upload the model.
#     """
#     # Save the model locally
#     joblib.dump(model, local_model_path)

#     # Upload the model to GCS
#     with fs.open(gcs_model_path, "wb") as f:
#         joblib.dump(model, f)


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


def upload_artifact(local_artifact_path, gcs_artifact_path):
    """
    Uploads the artifact to GCS.

    Parameters:
    local_artifact_path (str): The local path to the artifact.
    gcs_artifact_path (str): The GCS path to upload the artifact.
    """

    storage_client = storage.Client()
    bucket = storage_client.bucket("stock_price_prediction_dataset")
    blob = bucket.blob(gcs_artifact_path)
    blob.upload_from_filename(local_artifact_path)
    return True
