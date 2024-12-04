import pandas as pd
import numpy as np
import yfinance as yf
import requests
import glob
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
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
from dags.src.correlation import plot_correlation_matrix, removing_correlated_variables
from dags.src.lagged_features import add_lagged_features
from dags.src.feature_interactions import add_feature_interactions
from dags.src.technical_indicators import add_technical_indicators


def scaler(data: pd.DataFrame, mean=None, variance=None):
    """
    if training data, mean and variance should be None
    if test data, mean and variance should be provided, calculated from training data
    """
    logging.info("Starting data scaling")
    scaler = StandardScaler()

    # Split features and target
    X = data.drop(columns=["close", "date"])
    y = data["close"]
    date = data["date"]
    # split date
    train_date = date.iloc[: -int(len(date) * 0.1)]
    test_date = date.iloc[-int(len(date) * 0.1) :]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False, random_state=2024)

    scaler = scaler.fit(X_train)
    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    # add date and close column back to the scaled data
    scaled_data_train = pd.concat(
        [
            train_date.reset_index(drop=True),
            pd.DataFrame(scaled_X_train, columns=X.columns),
            y_train.reset_index(drop=True),
        ],
        axis=1,
    )

    scaled_data_test = pd.concat(
        [
            test_date.reset_index(drop=True),
            pd.DataFrame(scaled_X_test, columns=X.columns),
            y_test.reset_index(drop=True),
        ],
        axis=1,
    )

    final_scaled_data = pd.concat([scaled_data_train, scaled_data_test], axis=0)

    logging.info(f"Scaling completed. Scaled data shape: {final_scaled_data.shape}")
    # save dataset
    # scaled_data_train.to_csv("pipeline/airflow/dags/data/scaled_data_train.csv", index=False)
    # scaled_data_test.to_csv("pipeline/airflow/dags/data/scaled_data_test.csv", index=False)
    # final_scaled_data.to_csv("pipeline/airflow/dags/data/final_dataset_for_modeling.csv", index=False)

    all_data = {
        "scaled_data_train": scaled_data_train,
        "scaled_data_test": scaled_data_test,
        "final_scaled_data": final_scaled_data,
    }
    return all_data


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    data = merge_data(ticker_symbol)
    data = convert_type_of_columns(data)
    filtered_data = keep_latest_data(data, 10)
    removed_weekend_data = remove_weekends(filtered_data)
    filled_data = fill_missing_values(removed_weekend_data)
    removed_correlated_data = removing_correlated_variables(filled_data)
    lagged_data = add_lagged_features(removed_correlated_data)
    interaction_data = add_feature_interactions(lagged_data)
    technical_data = add_technical_indicators(interaction_data)
    scaled_data = scaler(technical_data)
    print(scaled_data)
