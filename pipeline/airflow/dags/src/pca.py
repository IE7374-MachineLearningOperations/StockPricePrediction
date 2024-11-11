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
from sklearn.decomposition import PCA
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Setting matplotlib logging level to suppress debug messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)

parent_path = os.path.abspath(os.path.dirname(__file__))

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
from dags.src.scaler import scaler


def apply_pca(data: pd.DataFrame, variance_threshold=0.95):
    """
    Standardizes the data and applies PCA for dimensionality reduction.
    Keeps enough components to explain the specified variance threshold.

    Returns:
    - X_pca: Array with PCA-transformed data.
    - explained_variance: Array of explained variance ratios by each PCA component.
    - n_components: Number of PCA components selected based on the variance threshold.
    """
    logging.info(f"Applying PCA with variance threshold: {variance_threshold}")
    data_pca = data.drop(columns=["date"])
    logging.debug(f"Shape of data for PCA: {data_pca.shape}")

    # Apply PCA
    pca = PCA(n_components=variance_threshold)
    reduced_data = pca.fit_transform(data_pca)
    
    # Check the explained variance by each component
    explained_variance = pca.explained_variance_ratio_
    n_components = pca.n_components_

    logging.info(f"PCA completed. Number of components: {n_components}")
    logging.debug(f"Explained variance by component: {explained_variance}")

    return reduced_data


def visualize_pca_components(data: pd.DataFrame, variance_threshold=0.95):
    """
    Visualizes the PCA components in a 2D plot.

    Returns:
    - None
    """
    logging.info("Visualizing PCA components")
    reduced_data = apply_pca(data, variance_threshold)[0]
    
    # Plot PCA components
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Components")
    
    # Ensure the assets directory exists
    assets_dir = "../../assets"
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
        logging.info(f"Created directory: {assets_dir}")
    
    plt.savefig(f"{assets_dir}/pca_components.png")
    logging.info(f"PCA components plot saved to {assets_dir}/pca_components.png")
    plt.show()


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

        scaled_data = scaler(technical_data)[0]
        logging.info(f"Data scaled. Shape: {scaled_data.shape}")

        visualize_pca_components(scaled_data, variance_threshold=0.95)
        logging.info("PCA visualization completed")

    except Exception as e:
        logging.error(f"An error occurred during data processing: {str(e)}")
        raise

    logging.info("Script execution completed")