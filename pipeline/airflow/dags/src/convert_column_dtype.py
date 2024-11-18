import pandas as pd
import yfinance as yf
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


def convert_type_of_columns(data: pd.DataFrame) -> pd.DataFrame:
    # convert 'date' to datetime
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    logging.info("Date column converted to datetime")

    # convert all numerical columns to float
    object_columns = data.select_dtypes(include=["object"]).columns

    for col in object_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        logging.debug(f"Column {col} converted to numeric")
    if data.empty:
        logging.error("Data conversion failed, resulting DataFrame is empty")
    else:
        logging.info("All columns converted successfully")

    return data


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    data = merge_data(ticker_symbol)
    data = convert_type_of_columns(data)
    print(data.dtypes)