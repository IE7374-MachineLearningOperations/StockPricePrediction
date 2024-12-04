import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
import glob
from datetime import datetime
import sys
import os
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

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


def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:

    logging.info("Start of removing and filling missing (NaN) values")

    # Dropping columns with all NaNs
    initial_columns = set(data.columns)
    data = data.drop(columns=[col for col in data.columns if data[col].isna().all()])
    dropped_columns = initial_columns - set(data.columns)
    if dropped_columns:
        logging.info(f"Dropped columns with all NaN values: {dropped_columns}")

    data.fillna(method="bfill", inplace=True)
    data.fillna(method="ffill", inplace=True)

    remaining_na = data.isna().sum().sum()
    if remaining_na > 0:
        logging.warning(f"There are still {remaining_na} missing values after filling")
    else:
        logging.info("Completed removing missing values. No more NaN values in the data")

    return data

    # return data


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    data = merge_data(ticker_symbol)
    data = convert_type_of_columns(data)
    filtered_data = keep_latest_data(data, 10)
    removed_weekend_data = remove_weekends(filtered_data)
    filled_data = fill_missing_values(removed_weekend_data)
    print(filled_data)
