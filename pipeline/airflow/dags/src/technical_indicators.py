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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Setting matplotlib logging level to suppress debug messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)

parent_path = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(os.path.dirname(os.path.dirname(parent_path)))

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


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to the DataFrame, including:
    - Relative Strength Index (RSI)
    - Moving Average Convergence Divergence (MACD)
    - Bollinger Bands

    Returns:
    - DataFrame with additional technical indicators.
    """

    logging.info("Starting to add technical indicators")

    # 1. Relative Strength Index (RSI) - 14-day RSI
    window_length = 14
    delta = data["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
    rs = gain / (loss + 1e-5)  # Adding small value to avoid division by zero
    data["RSI"] = 100 - (100 / (1 + rs))
    logging.debug("RSI calculated")

    # 2. Moving Average Convergence Divergence (MACD)
    short_window = 12
    long_window = 26
    signal_window = 9
    data["MACD"] = (
        data["close"].ewm(span=short_window, adjust=False).mean()
        - data["close"].ewm(span=long_window, adjust=False).mean()
    )
    data["MACD_signal"] = data["MACD"].ewm(span=signal_window, adjust=False).mean()
    logging.debug("MACD calculated")

    # 3. Bollinger Bands (20-day window, 2 standard deviations)
    data["MA20"] = data["close"].rolling(window=20).mean()
    data["BB_upper"] = data["MA20"] + (data["close"].rolling(window=20).std() * 2)
    data["BB_lower"] = data["MA20"] - (data["close"].rolling(window=20).std() * 2)
    logging.debug("Bollinger Bands calculated")

    # Drop NaN values introduced by rolling calculations
    original_shape = data.shape
    data = data.dropna()
    logging.info(
        f"Dropped rows with NaN values. Rows before: {original_shape[0]}, Rows after: {data.shape[0]}"
    )

    logging.info(f"Technical indicators added. New shape: {data.shape}")
    return data


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
        logging.info(f"Technical indicators added. Final shape: {technical_data.shape}")

        print(technical_data)
        logging.info("Data processing completed successfully")

    except Exception as e:
        logging.error(f"An error occurred during data processing: {str(e)}")
        raise

    # Optional: Save the final DataFrame to a file
    try:
        if not os.path.exists("artifacts"):
            os.makedirs("artifacts")
            logging.info("Created artifacts directory")

        technical_data.to_csv("artifacts/processed_data_with_technical_indicators.csv", index=False)
        logging.info("Processed data saved to artifacts/processed_data_with_technical_indicators.csv")
    except Exception as e:
        logging.error(f"Failed to save processed data: {str(e)}")

    logging.info("Script execution completed")
