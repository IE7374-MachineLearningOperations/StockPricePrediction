import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import lime
import lime.lime_tabular
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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
from dags.src.models.model_utils import save_and_upload_model, upload_artifact, split_time_series_data


# Hyperparameter tuning for Support Vector Machine (SVM)
def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        "C": [1, 10, 100],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf", "linear"],
        "epsilon": [0.1, 0.2, 0.5],
    }

    svr = SVR()
    grid_search = GridSearchCV(
        estimator=svr, param_grid=param_grid, cv=3, scoring="neg_mean_squared_error", verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_params_


def hyperparameter_sensitivity(X_train, y_train):
    param_ranges = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.1, 1],
        "epsilon": [0.01, 0.1, 0.5, 1],
    }

    for param, values in param_ranges.items():
        scores = []
        for value in values:
            if param == "gamma" and value in ["scale", "auto"]:
                svr = SVR(kernel="rbf", **{param: value})
            else:
                svr = SVR(kernel="rbf", **{param: float(value)})
            score = -np.mean(cross_val_score(svr, X_train, y_train, cv=3, scoring="neg_mean_squared_error"))
            scores.append(score)

        plt.figure(figsize=(10, 6))
        plt.plot(values, scores, "bo-")
        plt.xlabel(param)
        plt.ylabel("Mean Squared Error")
        plt.title(f"Hyperparameter Sensitivity: {param}")
        plt.xscale("log" if param != "gamma" else "linear")
        plt.show()


def feature_importance_analysis(model, X_train, X_test, feature_names):
    # Permutation Importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    sorted_idx = perm_importance.importances_mean.argsort()
    top_10_idx = sorted_idx[-10:]

    plt.figure(figsize=(10, 6))
    plt.barh(range(10), perm_importance.importances_mean[top_10_idx])
    plt.yticks(range(10), [feature_names[i] for i in top_10_idx])
    plt.title("Top 10 Features - Permutation Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    # LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values, feature_names=feature_names, mode="regression"
    )
    exp = explainer.explain_instance(X_test.iloc[0], model.predict, num_features=10)
    exp.as_pyplot_figure()
    plt.title("Top 10 Features - LIME")
    plt.tight_layout()
    plt.show()


# Train the model with the best parameters
def train_svm(model, data, target_column="close", test_size=0.1, param_grid=None, tscv=None):
    X_train, _, y_train, _ = split_time_series_data(
        data, target_column="close", test_size=0.1, random_state=2024
    )
    pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("model", model)]  # Handle NaNs in the dataset
    )
    # train svr model
    if param_grid:
        search = GridSearchCV(
            pipeline, param_grid, cv=tscv, scoring="neg_mean_squared_error", error_score="raise"
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_  # average of valid datasets
        best_params = search.best_params_

        return best_model, best_params


# Evaluate the model
def predict_svm(data, target_column="close", test_size=0.1, val_size=0.1, random_state=2024):
    # Evaluate on validation set
    # y_val_pred = model.predict(X_val)
    # svm_val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    # svm_val_mae = mean_absolute_error(y_val, y_val_pred)
    # svm_val_r2 = r2_score(y_val, y_val_pred)

    # print(f"Validation RMSE: {svm_val_rmse:.4f}, MAE: {svm_val_mae:.4f}, R²: {svm_val_r2:.4f}")

    pred_metrics = {}

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_time_series_data(data, target_column, test_size, random_state)

    # Time Series Cross-Validation for train sets (train and validation)
    tscv = TimeSeriesSplit(n_splits=5)

    # Define models and hyperparameters for hyperparameter tuning
    svr_best_model = {}
    model_name = "Support Vector Regression"
    model = SVR()
    param_grid = {
        "model__C": [10, 100],
        "model__gamma": ["scale"],
        "model__epsilon": [0.01],
        "model__kernel": ["linear"],
    }
    print(f"\nTraining {model_name} model...")
    best_model, _ = train_svm(
        model, data, target_column="close", test_size=0.1, param_grid=param_grid, tscv=tscv
    )
    # Evaluate on test set
    # y_test = model.predict(X_test)
    # svm_test_rmse = np.sqrt(mean_squared_error(y_test, y_test))
    # svm_test_mae = mean_absolute_error(y_test, y_test)
    # svm_test_r2 = r2_score(y_test, y_test)
    y_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    # print(f"Test RMSE: {svm_test_rmse:.4f}, MAE: {svm_test_mae:.4f}, R²: {svm_test_r2:.4f}")
    # Store results in the results dictionary (including validation and test results)
    pred_metrics[model_name] = {
        "test_MSE": test_mse,
        "test_RMSE": test_rmse,
        "test_MAE": test_mae,
        "test_R2": test_r2,
    }

    # Store the trained model for plotting later
    svr_best_model[model_name] = best_model
    # Plot Actual vs Predicted (Test Set) with reduced x-axis date labels
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_test.index, y_test.values, label="Actual", color="blue")
    # plt.plot(y_test.index, y_test, label="Predicted", color="red")
    # plt.title("Stock Price Prediction - Actual vs Predicted (SVM)")
    # plt.xlabel("Date")
    # plt.ylabel("Stock Price")
    # plt.legend()

    # Set fewer date ticks at regular intervals
    # ax = plt.gca()  # Get current axis
    # tick_frequency = len(y_test) // 10  # Show one tick every 10% of the data, adjust as needed
    # ax.set_xticks(y_test.index[::tick_frequency])

    # # Rotate the x-axis labels vertically for better readability
    # plt.xticks(rotation=90, size=8)

    # plt.tight_layout()
    # plt.show()

    # Return metrics as a dictionary for later use
    return pred_metrics, svr_best_model, y_test, y_pred


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
    pred_metrics, lr_best_model, y_test, y_pred = predict_svm(scaled_data)
    print(pred_metrics, lr_best_model)
