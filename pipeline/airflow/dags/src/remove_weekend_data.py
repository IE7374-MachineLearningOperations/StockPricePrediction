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


def remove_weekends(data: pd.DataFrame) -> pd.DataFrame:

    logging.info("Removing weekend data")

    # Removing weekends (Saturday = 5, Sunday = 6)
    removed_weekend_data = data[~data["date"].dt.weekday.isin([5, 6])]

    num_removed = len(data) - len(removed_weekend_data)
    logging.info(f"Removed {num_removed} rows of weekend dates")

    if removed_weekend_data.empty:
        logging.error("Resulting DataFrame is empty")
    else:
        logging.info("Weekend dates data removed successfully")

    return removed_weekend_data


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    data = merge_data(ticker_symbol)
    data = convert_type_of_columns(data)
    filtered_data = keep_latest_data(data, 10)
    removed_weekend_data = remove_weekends(filtered_data)
    print(removed_weekend_data)