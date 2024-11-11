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
    data_scaling = data.drop(columns=["date"])

    if mean is not None and variance is not None:
        logging.info("Using provided mean and variance for scaling")
        scaler = scaler.fit(data_scaling)
        scaler.mean_ = mean
        scaler.var_ = variance
        scaled_data = scaler.transform(data_scaling)
    else:
        logging.info("Fitting and transforming data")
        scaled_data = scaler.fit_transform(data_scaling)

    # add date column back to the scaled data
    final_scaled_data = pd.concat(
        [data["date"].reset_index(drop=True), pd.DataFrame(scaled_data, columns=data_scaling.columns)], axis=1
    )

    logging.info(f"Scaling completed. Scaled data shape: {final_scaled_data.shape}")

    return final_scaled_data


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    logging.info(f"Starting data processing for {ticker_symbol}")

    try:
        data = merge_data(ticker_symbol)
        logging.info(f"Data merged. Shape: {data.shape}")

        data = convert_type_of_columns(data)
        logging.info("Column types converted")

        filtered_data = keep_latest_data(data, 10)
        logging.info(f"Kept latest data. New shape: {filtered_data.shape}")

        removed_weekend_data = remove_weekends(filtered_data)
        logging.info(f"Weekend data removed. New shape: {removed_weekend_data.shape}")

        filled_data = fill_missing_values(removed_weekend_data)
        logging.info(f"Missing values filled. Shape: {filled_data.shape}")

        removed_correlated_data = removing_correlated_variables(filled_data)
        logging.info(f"Correlated variables removed. New shape: {removed_correlated_data.shape}")

        lagged_data = add_lagged_features(removed_correlated_data)
        logging.info(f"Lagged features added. New shape: {lagged_data.shape}")

        interaction_data = add_feature_interactions(lagged_data)
        logging.info(f"Feature interactions added. New shape: {interaction_data.shape}")

        technical_data = add_technical_indicators(interaction_data)
        logging.info(f"Technical indicators added. New shape: {technical_data.shape}")

        scaled_data, mean, variance = scaler(technical_data)
        logging.info(f"Data scaled. Final shape: {scaled_data.shape}")

        print(scaled_data)
        logging.info("Data processing completed successfully")

    except Exception as e:
        logging.error(f"An error occurred during data processing: {str(e)}")
        raise

    # Optional: Save the final DataFrame to a file
    try:
        if not os.path.exists("artifacts"):
            os.makedirs("artifacts")
            logging.info("Created artifacts directory")

        scaled_data.to_csv("artifacts/scaled_data.csv", index=False)
        logging.info("Scaled data saved to artifacts/scaled_data.csv")

        np.save("artifacts/scaler_mean.npy", mean)
        np.save("artifacts/scaler_variance.npy", variance)
        logging.info("Scaler mean and variance saved to artifacts/")
    except Exception as e:
        logging.error(f"Failed to save processed data: {str(e)}")

    logging.info("Script execution completed")
