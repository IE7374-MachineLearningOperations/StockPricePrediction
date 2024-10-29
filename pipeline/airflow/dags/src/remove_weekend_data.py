import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
import glob
from datetime import datetime
import sys
import os

parent_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(parent_path))))
sys.path.append(root_path)
from pipeline.airflow.dags.src.download_data import (
    get_yfinance_data,
    get_fama_french_data,
    get_ads_index_data,
    get_sp500_data,
    get_fred_data,
    merge_data,
)
from pipeline.airflow.dags.src.convert_column_dtype import convert_type_of_columns
from pipeline.airflow.dags.src.keep_latest_data import keep_latest_data


def remove_weekends(data: pd.DataFrame) -> pd.DataFrame:

    # Removing weekends (Saturday = 5, Sunday = 6)
    removed_weekend_data = data[~data["date"].dt.weekday.isin([5, 6])]

    return removed_weekend_data


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    data = merge_data(ticker_symbol)
    data = convert_type_of_columns(data)
    filtered_data = keep_latest_data(data, 10)
    removed_weekend_data = remove_weekends(filtered_data)
    print(removed_weekend_data)
