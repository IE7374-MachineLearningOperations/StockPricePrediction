import numpy as np
import pandas as pd
import sys
import os
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# from xgboost import XGBRegressor
import xgboost as xg


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
from dags.src.correlation import removing_correlated_variables
from dags.src.lagged_features import add_lagged_features
from dags.src.feature_interactions import add_feature_interactions
from dags.src.technical_indicators import add_technical_indicators
from dags.src.scaler import scaler
from dags.src.upload_blob import upload_blob
from dags.src.models.model_utils import save_and_upload_model


# Define the data preparation function
def prepare_data(data, test_size=0.2):
    X = data.drop(["close", "date"], axis=1)
    y = data["close"]

    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def train_xgboost_with_metrics(data, test_size=0.2):
    # log the start of the training process
    logging.info("Starting XGBoost model training")

    X_train, X_test, y_train, y_test = prepare_data(data, test_size)
    # Initialize the XGBoost regressor
    xgb_model = xg.XGBRegressor(objective="reg:squarederror", random_state=42)

    # Define a hyperparameter grid for tuning
    param_grid = {
        "n_estimators": [5],
        "max_depth": [3, 4],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.85],
        "colsample_bytree": [0.85],
    }

    # Set up GridSearchCV with 5-fold cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=-1,
    )

    # Fit the model to the training data
    grid_search.fit(X_train, y_train)

    # Get the best estimator
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred_xgb = best_model.predict(X_test)

    # Calculate evaluation metrics
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    mape_xgb = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100

    # Print best parameters and evaluation metrics
    logging.info("Best Parameters and metrics for XGBoost model")
    logging.info("Best Parameters: %s", grid_search.best_params_)
    logging.info("RMSE - XGBoost: %f", rmse_xgb)
    logging.info("MAE - XGBoost: %f", mae_xgb)
    logging.info("MAPE - XGBoost: %f", mape_xgb)

    logging.info(f"Uploading the best model LSTM to Google Cloud Storage")

    folder = "artifacts"
    if not os.path.exists(folder):
        os.makedirs(folder)
    output_path = f"{folder}/XGBoost.pkl"
    gcs_model_path = "model_checkpoints/XGBoost.pkl"
    save_and_upload_model(model=best_model, local_model_path=output_path, gcs_model_path=gcs_model_path)

    # Return the best model and metrics
    return best_model, rmse_xgb, mae_xgb, mape_xgb


if __name__ == "__main__":
    ticker_symbol = "GOOGL"
    data = merge_data(ticker_symbol)
    data = convert_type_of_columns(data)
    filtered_data = keep_latest_data(data, 10)
    removed_weekend_data = remove_weekends(filtered_data)
    filled_data = fill_missing_values(removed_weekend_data)
    removed_correlated_data = removing_correlated_variables(filled_data)
    lagged_data = add_lagged_features(removed_correlated_data)
    feature_interactions_data = add_feature_interactions(lagged_data)
    technical_indicators_data = add_technical_indicators(feature_interactions_data)
    scaled_data = scaler(technical_indicators_data)
    best_model, rmse, mae, mape = train_xgboost_with_metrics(scaled_data)
    print("Model training complete.")
    print(best_model, rmse, mae, mape)
