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


def add_feature_interactions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds interaction features between specified columns in the DataFrame.
    Includes product and ratio interactions for 'close' with 'SP500' and 'VIXCLS',
    as well as interactions between 'SP500', 'VIXCLS'

    Returns:
    - DataFrame with additional interaction features.
    """
    logging.info("Starting to add feature interactions")

    # Interaction between 'close' price and 'SP500' (product and ratio)
    data["close_SP500_product"] = data["close"] * data["SP500"]
    data["close_SP500_ratio"] = data["close"] / (data["SP500"] + 1e-5)  # Avoid division by zero
    logging.debug("Added close-SP500 interactions")

    # Interaction between 'close' price and 'VIXCLS' (volatility)
    data["close_VIXCLS_product"] = data["close"] * data["VIXCLS"]
    data["close_VIXCLS_ratio"] = data["close"] / (data["VIXCLS"] + 1e-5)
    logging.debug("Added close-VIXCLS interactions")

    # Interaction between 'SP500' and 'VIXCLS'
    data["SP500_VIXCLS_product"] = data["SP500"] * data["VIXCLS"]
    data["SP500_VIXCLS_ratio"] = data["SP500"] / (data["VIXCLS"] + 1e-5)
    logging.debug("Added SP500-VIXCLS interactions")

    logging.info(f"Added 6 new interaction features. New shape: {data.shape}")
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
        logging.info(f"Feature interactions added. Final shape: {interaction_data.shape}")

        print(interaction_data)
        logging.info("Data processing completed successfully")

    except Exception as e:
        logging.error(f"An error occurred during data processing: {str(e)}")
        raise

    # Optional: Save the final DataFrame to a file
    try:
        if not os.path.exists("artifacts"):
            os.makedirs("artifacts")
            logging.info("Created artifacts directory")

        interaction_data.to_csv("artifacts/processed_data.csv", index=False)
        logging.info("Processed data saved to artifacts/processed_data.csv")
    except Exception as e:
        logging.error(f"Failed to save processed data: {str(e)}")

    logging.info("Script execution completed")
