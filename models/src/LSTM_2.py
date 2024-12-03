import numpy as np
import pandas as pd
import sys
import os
import torch
import torch.nn as nn
import itertools
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit


import logging
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# import wandb  ## TODO
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
from dags.src.models.model_utils import save_and_upload_model, split_time_series_data


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        h_dropout = self.dropout(h_lstm[:, -1, :])
        out = self.fc(h_dropout)
        return out


class LSTMEstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        target_column="close",
        timesteps=10,
        hidden_size=50,
        dropout_rate=0.2,
        optimizer_type="adam",
        batch_size=32,
        epochs=20,
    ):
        self.target_column = target_column
        self.timesteps = timesteps
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.optimizer_type = optimizer_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.scaler = None

    def _preprocess_data(self, data):
        X = data.drop([self.target_column, "date"], axis=1)
        y = data[self.target_column]

        X_reshaped, y_reshaped = [], []
        for i in range(self.timesteps, len(X)):
            X_reshaped.append(X[i - self.timesteps : i])
            y_reshaped.append(y[i])

        X_reshaped, y_reshaped = np.array(X_reshaped), np.array(y_reshaped)

        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, y_reshaped, test_size=0.1, random_state=2024, shuffle=False
        )
        return (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
            self.scaler,
        )

    def fit(self, X, y):
        data = pd.concat([X, y], axis=1)
        X_train, X_test, y_train, y_test, self.scaler = self._preprocess_data(data)
        input_size = X_train.shape[2]
        self.model = LSTMModel(input_size, self.hidden_size, self.dropout_rate)

        criterion = nn.MSELoss()
        optimizer = (
            torch.optim.Adam(self.model.parameters())
            if self.optimizer_type == "adam"
            else torch.optim.SGD(self.model.parameters(), lr=0.01)
        )

        self.model.train()
        for epoch in range(self.epochs):
            permutation = torch.randperm(X_train.size(0))
            for i in range(0, X_train.size(0), self.batch_size):
                indices = permutation[i : i + self.batch_size]
                batch_X, batch_y = X_train[indices], y_train[indices]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y.view(-1, 1))
                loss.backward()
                optimizer.step()

        return self

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            X_train, X_test, y_train, y_test, self.scaler = self._preprocess_data(self.data)
            y_pred = self.model(X_test).numpy()
            y_pred_rescaled = self.scaler.inverse_transform(y_pred)
        return y_pred_rescaled


# train lstm with grid search
def train_lstm(data, target_column="close", test_size=0.1, timesteps=10):
    X_train, X_test, y_train, y_test, scaler = LSTMEstimator(data)._preprocess_data(data)
    pipeline = Pipeline([("scaler", StandardScaler()), ("lstm", LSTMEstimator(data))])

    param_grid = {
        "lstm__hidden_size": [50, 100],
        "lstm__dropout_rate": [0.2, 0.5],
        "lstm__optimizer_type": ["adam", "sgd"],
        "lstm__batch_size": [32, 64],
        "lstm__epochs": [20, 50],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring="neg_mean_squared_error")
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    print("Best parameters found: ", best_params)

    return best_model, best_params


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
    best_model, best_params = train_lstm(scaled_data)
    print(best_model, best_params)
