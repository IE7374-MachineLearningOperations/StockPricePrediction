import os
from google.cloud import storage
import pandas as pd
import gcsfs
import sys

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


abs_path = os.path.abspath(__file__)
dir = os.path.dirname(abs_path)
dir = os.path.dirname(dir)
path = os.path.join(dir, "service_key_gcs.json")

if os.path.exists(path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
    storage_client = storage.Client()
else:
    storage_client = None
    logging.warning("------- Service key not found!")

"""
Upload a file to the bucket
"""


def upload_blob(data, gcs_file_path: str = None):
    # """Uploads a file to the bucket."""

    # if gcs_file_path is None:
    #     gcs_file_path = "gs://stock_price_prediction_dataset/Data/pipeline/airflow/dags/data/final_dataset_for_modeling.csv"

    # data.to_csv(gcs_file_path, index=False)
    """
    upload_from_string()
    upload_from_file()
    upload_from_filename()
    """
    if gcs_file_path is None:
        gcs_file_path = "gs://stock_price_prediction_dataset/Data/pipeline/airflow/dags/data/final_dataset_for_modeling.csv"

    storage_client = storage.Client()
    bucket = storage_client.bucket("stock_price_prediction_dataset")
    blob = bucket.blob(gcs_file_path)
    blob.upload_from_string(data.to_csv(index=False), "text/csv")
    return True


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    data = merge_data(ticker_symbol)
    data = convert_type_of_columns(data)
    filtered_data = keep_latest_data(data, 10)
    removed_weekend_data = remove_weekends(filtered_data)
    filled_data = fill_missing_values(removed_weekend_data)
    removed_correlated_data = removing_correlated_variables(filled_data)
    lagged_data = add_lagged_features(removed_correlated_data)
    feature_interactions_data = add_feature_interactions(lagged_data)
    technical_indicators_data = add_technical_indicators(feature_interactions_data)
    scaled_data = scaler(technical_indicators_data)
    upload_blob(scaled_data)
    print("done!!")
