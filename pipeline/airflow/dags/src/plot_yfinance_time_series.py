import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
import glob
from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt
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
from dags.src.handle_missing import fill_missing_values


def plot_yfinance_time_series(data: pd.DataFrame):

    logging.info("Plotting time series line chart for yfianance columns")

    data.set_index("date", inplace=True)
    logging.debug("Set 'date' as index")

    yfinance_columns = ["open", "high", "low", "close", "volume"]
    yfinance_data = data[yfinance_columns]
    logging.debug(f"Filtered data to include only yfinance columns: {yfinance_columns}")

    num_rows = yfinance_data.shape[1]

    fig, axs = plt.subplots(num_rows, 1, figsize=(15, 5 * num_rows))
    axs = axs.flatten()

    for i, column in enumerate(yfinance_data.columns):
        axs[i].plot(yfinance_data.index, yfinance_data[column], label=column)
        axs[i].set_title(column)
        axs[i].set_xlabel("Date")
        axs[i].set_ylabel(column)
        axs[i].legend()
        logging.debug(f"Created subplot for {column}")

    plt.tight_layout()

    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
        logging.info("Created 'artifacts' directory")

    plt.savefig("artifacts/yfinance_time_series.png")
    logging.info("Saved plot in artifacts'")

    logging.info("Finished plotting yfinance time series")


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    data = merge_data(ticker_symbol)
    data = convert_type_of_columns(data)
    filtered_data = keep_latest_data(data, 10)
    removed_weekend_data = remove_weekends(filtered_data)
    filled_data = fill_missing_values(removed_weekend_data)
    plot_yfinance_time_series(filled_data)