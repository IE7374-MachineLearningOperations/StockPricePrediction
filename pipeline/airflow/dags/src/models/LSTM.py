import numpy as np
import pandas as pd
import sys
import os
import torch
import torch.nn as nn
import itertools

import logging
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import wandb
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


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


# Define the LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take the last output for the final layer
        x = self.fc(x)
        return x


# Function to preprocess the data
def preprocess_data(df, timesteps=10):
    scaler = MinMaxScaler()
    X = df.drop(["close", "date"], axis=1)
    y = df["close"]

    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

    X_reshaped, y_reshaped = [], []
    for i in range(timesteps, len(X_scaled)):
        X_reshaped.append(X_scaled[i - timesteps : i])
        y_reshaped.append(y_scaled[i])

    X_reshaped, y_reshaped = np.array(X_reshaped), np.array(y_reshaped)

    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y_reshaped, test_size=0.2, random_state=42, shuffle=False
    )

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        scaler,
    )


# Function to train and evaluate the LSTM model
def train_and_evaluate_lstm(
    df, timesteps=10, hidden_size=50, dropout_rate=0.2, optimizer_type="adam", batch_size=32, epochs=20
):
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    config = {"timesteps": timesteps, "hidden_size": hidden_size, "dropout_rate": dropout_rate}
    logger = wandb.init(project="stock-price-prediction", name=f"experiment_{cur_time}", config=config)

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df, timesteps)

    input_size = X_train.shape[2]
    model = LSTMModel(input_size, hidden_size, dropout_rate)

    criterion = nn.MSELoss()
    optimizer = (
        torch.optim.Adam(model.parameters())
        if optimizer_type == "adam"
        else torch.optim.SGD(model.parameters(), lr=0.01)
    )

    # Training loop
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i : i + batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.view(-1, 1))
            loss.backward()
            optimizer.step()

            wandb.log({"epoch": epoch, "bid": i, "loss": loss.item()})

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy()

    # Inverse scaling of predictions and true values
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.view(-1, 1).numpy())

    # Calculate error metrics
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100

    ## logging the results
    logger.log({"RMSE": rmse, "MAE": mae, "MAPE": mape})

    logging.info("Results....")
    logging.info("RMSE - LSTM: %f", rmse)
    logging.info("MAE - LSTM: %f", mae)
    logging.info("MAPE - LSTM: %f", mape)

    wandb.finish()

    return model, rmse


def grid_search_lstm(df, param_grid=None):
    if param_grid is None:
        param_grid = {
            "timesteps": [10],
            "hidden_size": [5, 10],
            "dropout_rate": [0.2, 0.3],
            "optimizer_type": ["adam"],
            "batch_size": [32],
            "epochs": [2],
        }

    logging.info("Starting grid search for LSTM model")
    best_params = None
    best_rmse = float("inf")
    best_model = None

    folder = "artifacts"
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Iterate over all combinations of hyperparameters
    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        logging.info(f"Testing parameters: {param_dict}")

        model, rmse = train_and_evaluate_lstm(
            df,
            timesteps=param_dict["timesteps"],
            hidden_size=param_dict["hidden_size"],
            dropout_rate=param_dict["dropout_rate"],
            optimizer_type=param_dict["optimizer_type"],
            batch_size=param_dict["batch_size"],
            epochs=param_dict["epochs"],
        )

        # hyperparameter_sensitivity(df, param_dict)

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = param_dict
            best_model = model

    logging.info(f"Best model for LSTM: {best_model}")
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best RMSE: {best_rmse}")

    logging.info(f"Uploading the best model LSTM to Google Cloud Storage")
    output_path = f"{folder}/LSTM.pkl"
    gcs_model_path = "model_checkpoints/LSTM.pkl"
    save_and_upload_model(model=model, local_model_path=output_path, gcs_model_path=gcs_model_path)

    return best_model, best_params, best_rmse


def hyperparameter_sensitivity(df, param_ranges):
    results = []
    for param, values in param_ranges.items():
        param_results = []
        for value in values:
            if param == "units":
                _, rmse = train_and_evaluate_lstm(df, units=value)
            elif param == "dropout_rate":
                _, rmse = train_and_evaluate_lstm(df, dropout_rate=value)
            elif param == "optimizer":
                _, rmse = train_and_evaluate_lstm(df, optimizer=value)
            param_results.append((value, rmse))
        results.append((param, param_results))

    # Plot results
    for param, param_results in results:
        values, rmses = zip(*param_results)
        plt.figure(figsize=(10, 6))
        plt.plot(values, rmses, marker="o")
        plt.title(f"Sensitivity Analysis: {param}")
        plt.xlabel(param)
        plt.ylabel("RMSE")

        # Ensure the assets directory exists
        if not os.path.exists("artifacts"):
            os.makedirs("artifacts")
            logging.info(f"Created directory: artifacts")

        plt.savefig(f"artifacts/Sensitivity Analysis - {param}.png")
        logging.info(f"Sensitivity Analysis plot saved to artifacts/Sensitivity Analysis - {param}.png")


def feature_importance(model, X_test):
    # Use SHAP for feature importance
    explainer = shap.DeepExplainer(model, X_test[:100])
    shap_values = explainer.shap_values(X_test[:100])

    # Plot feature importance
    shap.summary_plot(shap_values[0], X_test[:100], plot_type="bar", show=False)


if __name__ == "__main__":
    # Hyperparameters
    timesteps = 10
    units = 50
    dropout_rate = 0.2
    optimizer = "adam"
    batch_size = 32
    epochs = 20

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
    param_grid = {
        "timesteps": [10],
        "hidden_size": [5, 10],
        "dropout_rate": [0.2, 0.3],
        "optimizer_type": ["adam"],
        "batch_size": [32],
        "epochs": [2],
    }

    best_model, best_params, best_rmse = grid_search_lstm(scaled_data, param_grid)
