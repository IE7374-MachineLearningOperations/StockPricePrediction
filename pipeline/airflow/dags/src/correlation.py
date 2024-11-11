import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
import glob
from datetime import datetime
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Setting matplotlib logging level to suppress debug messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)

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


def plot_correlation_matrix(data: pd.DataFrame):
    logging.info("Plotting correlation matrix")
    plt.figure(figsize=(25, 18))
    sns.heatmap(data.corr(), annot=True, fmt=".2f")
    plt.title("Correlation Matrix")

    # Creating artifacts directory
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
        logging.info("Created artifacts directory")

    plt.savefig("artifacts/correlation_matrix_original_dataset.png")
    logging.info("Correlation matrix plot for original dataset saved in artifacts")


def removing_correlated_variables(data: pd.DataFrame) -> pd.DataFrame:

    logging.info("Starting removal of correlated variables")
    df_corr = data.corr()
    NaN_columns = [col for col in df_corr.columns if df_corr[col].isna().all()]
    df__withoutNaN = data.drop(columns=NaN_columns)
    logging.info(f"NaN correlation/ 0 valued columns dropped: {NaN_columns}")

    corr_matrix_cleaned = df__withoutNaN.corr()
    columns_to_drop = set()

    for i in range(len(corr_matrix_cleaned.columns)):
        for j in range(i):
            if abs(corr_matrix_cleaned.iloc[i, j]) >= 0.95:
                colname = corr_matrix_cleaned.columns[i]
                if colname not in ["open", "high", "low", "close", "volume", "SP500"]:
                    columns_to_drop.add(colname)

    df_final = df__withoutNaN.drop(columns=columns_to_drop)
    logging.info(f"Highly correlated columns dropped: {columns_to_drop}")

    plt.figure(figsize=(25, 18))
    sns.heatmap(df_final.corr(), annot=True, fmt=".2f")
    plt.title("Correlation Matrix")

    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
        logging.info("Created artifacts directory")

    plt.savefig("artifacts/correlation_matrix_after_removing_correlated_features.png")
    logging.info("Correlation matrix plot after removing correlated features saved in artifacts")

    return df_final


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    data = merge_data(ticker_symbol)
    data = convert_type_of_columns(data)
    filtered_data = keep_latest_data(data, 10)
    removed_weekend_data = remove_weekends(filtered_data)
    filled_data = fill_missing_values(removed_weekend_data)
    removed_correlated_data = removing_correlated_variables(filled_data)
    print(removed_correlated_data)