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

# sys.path.append(os.path.abspath("pipeline/airflow"))
# sys.path.append(os.path.abspath("."))

# from dags.src.download_data import (
#     get_yfinance_data,
#     get_fama_french_data,
#     get_ads_index_data,
#     get_sp500_data,
#     get_fred_data,
#     merge_data,
# )
# from dags.src.convert_column_dtype import convert_type_of_columns
# from dags.src.keep_latest_data import keep_latest_data
# from dags.src.remove_weekend_data import remove_weekends
# from dags.src.handle_missing import fill_missing_values
# from dags.src.correlation import plot_correlation_matrix, removing_correlated_variables


def add_lagged_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering process:
    - Selects relevant columns
    - Adds lagged features (1, 3, 5 days) for 'close', 'open', 'high', and 'low'
    - Adds rolling mean and volatility features for 'close' over 5-day and 10-day windows
    - Drops rows with NaN values resulting from these transformations

    Returns:
    - DataFrame with engineered features.
    """
    logging.info("Starting to add lagged features")

    # Step 1: Define columns for lagged features
    columns_to_lag = ["close", "open", "high", "low"]
    logging.debug(f"Columns to lag: {columns_to_lag}")

    # Step 2: Add lagged features (1-day, 3-day, 5-day lags)
    for column in columns_to_lag:
        data[f"{column}_lag1"] = data[column].shift(1)
        data[f"{column}_lag3"] = data[column].shift(3)
        data[f"{column}_lag5"] = data[column].shift(5)
    logging.info("Lagged features added")

    # Step 3: Add rolling statistics (5-day and 10-day moving averages and volatility) for 'close'
    data["close_ma5"] = data["close"].rolling(window=5).mean()
    data["close_ma10"] = data["close"].rolling(window=10).mean()
    data["close_vol5"] = data["close"].rolling(window=5).std()
    data["close_vol10"] = data["close"].rolling(window=10).std()
    logging.info("Rolling statistics added")

    # Step 4: Drop rows with NaN values generated by lagging and rolling operations
    original_shape = data.shape
    data = data.dropna()
    logging.info(
        f"Dropped rows with NaN values. Rows before: {original_shape[0]}, Rows after: {data.shape[0]}"
    )

    logging.info(f"Lagged features added. New shape: {data.shape}")

    return data


if __name__ == "__main__":
    pass
    # ticker_symbol = "GOOGL"
    # logging.info(f"Starting data processing for {ticker_symbol}")

    # try:
    #     data = merge_data(ticker_symbol)
    #     logging.info(f"Data merged. Shape: {data.shape}")

    #     data = convert_type_of_columns(data)
    #     logging.info("Column types converted")

    #     filtered_data = keep_latest_data(data, 10)
    #     logging.info(f"Kept latest data. New shape: {filtered_data.shape}")

    #     removed_weekend_data = remove_weekends(filtered_data)
    #     logging.info(f"Weekend data removed. New shape: {removed_weekend_data.shape}")

    #     filled_data = fill_missing_values(removed_weekend_data)
    #     logging.info(f"Missing values filled. Shape: {filled_data.shape}")

    #     removed_correlated_data = removing_correlated_variables(filled_data)
    #     logging.info(f"Correlated variables removed. New shape: {removed_correlated_data.shape}")

    #     lagged_data = add_lagged_features(removed_correlated_data)
    #     logging.info(f"Lagged features added. Final shape: {lagged_data.shape}")

    #     print(lagged_data)
    #     logging.info("Data processing completed successfully")

    # except Exception as e:
    #     logging.error(f"An error occurred during data processing: {str(e)}")
    #     raise

    # # Optional: Save the final DataFrame to a file
    # try:
    #     if not os.path.exists("artifacts"):
    #         os.makedirs("artifacts")
    #         logging.info("Created artifacts directory")

    #     lagged_data.to_csv("artifacts/processed_data_with_lags.csv", index=False)
    #     logging.info("Processed data saved to artifacts/processed_data_with_lags.csv")
    # except Exception as e:
    #     logging.error(f"Failed to save processed data: {str(e)}")
    # logging.info("Script execution completed")
