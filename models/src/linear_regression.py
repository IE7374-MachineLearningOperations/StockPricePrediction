import numpy as np
import pandas as pd
import sys
import os
import logging
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import joblib
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
import plotly.graph_objs as go


def plot_actual_vs_predicted_interactive(model_name, date_test, y_pred, y_test):
    fig = go.Figure()

    # Add actual values
    fig.add_trace(
        go.Scatter(
            x=date_test,
            y=y_test.values.flatten(),
            mode="lines",
            name="Actual",
            line=dict(color="black", width=2),
            hovertemplate="Date: %{x}<br>Actual: %{y}<extra></extra>",
        )
    )

    # Add predicted values
    fig.add_trace(
        go.Scatter(
            x=date_test,
            y=y_pred.flatten(),
            mode="lines",
            name=f"Predicted ({model_name})",
            line=dict(dash="dash"),
            hovertemplate="Date: %{x}<br>Predicted: %{y}<extra></extra>",
        )
    )

    # Update layout
    fig.update_layout(
        title="Actual vs Predicted Values",
        xaxis_title="Date",
        yaxis_title="Close Value",
        legend=dict(x=0, y=1),
        hovermode="x unified",
    )
    # save the plot
    fig.write_html("artifacts/actual_vs_predicted_interactive.html")


def detect_bias(data_train):
    X_train = data_train.drop(columns=["date", "close"])
    y_train = data_train["close"]

    # split X_train to train and validation
    X_train = X_train.iloc[: -int(len(X_train) * 0.1)]
    y_train = y_train.iloc[: -int(len(y_train) * 0.1)]
    X_val = X_train.iloc[-int(len(X_train) * 0.1) :]
    y_val = y_train.iloc[-int(len(y_train) * 0.1) :]

    # Define features and target
    features = [col for col in data_train.columns if col != "close" and col != "date"]
    target = "close"

    # Add the target column to the training set for resampling
    train_data = X_train.copy()
    train_data["close"] = y_train
    # Add the target to the training features for resampling

    # Define the slice condition based on VIX values
    vix_median = pd.to_numeric(train_data["VIXCLS"], errors="coerce").median()
    high_vix_data = train_data[train_data["VIXCLS"] > vix_median]
    low_vix_data = train_data[train_data["VIXCLS"] <= vix_median]

    # Upsample high-error slice data
    high_vix_upsampled = resample(high_vix_data, replace=True, n_samples=len(low_vix_data), random_state=42)
    balanced_train_data = pd.concat([high_vix_upsampled, low_vix_data])

    # Separate features and target after balancing
    X_train_balanced = balanced_train_data.drop(columns=["close"])  # Drop the target from balanced data
    y_train_balanced = balanced_train_data["close"]  # Get the target values

    # Train the ElasticNet model on the balanced data
    model = Ridge(alpha=0.05)
    model.fit(X_train_balanced, y_train_balanced)

    # Predict and evaluate on the test set
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    logging.info(f"Linear Regression Test MSE with Resampling: {mse}")
    logging.info(f"Linear Regression Test MAE with Resampling: {mae}")

    # save mse and mae to a file
    with open("artifacts/bias_detection_metrics.txt", "w") as f:
        f.write(f"Linear Regression Test MSE with Resampling: {mse}")
        f.write(f"Linear Regression Test MAE with Resampling: {mae}")


def feature_importance_analysis(model, X_train, y_train):
    # Permutation feature importance
    perm_importance = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42)

    # Sort features by importance
    sorted_idx = perm_importance.importances_mean.argsort()

    # Select top 10 features
    top_10_idx = sorted_idx[-10:]

    plt.figure(figsize=(10, 6))
    plt.barh(range(10), perm_importance.importances_mean[top_10_idx])
    plt.yticks(range(10), X_train.columns[top_10_idx])
    plt.title("Top 10 Features - Permutation Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("artifacts/feature_importance.png")


# Function to train and evaluate models with hyperparameter tuning and time series split
def train_linear_regression(data_train, target_column="close", param_grid=None, tscv=None):
    """
    Train a model on the given data and evaluate it using TimeSeriesSplit cross-validation.
    Optionally perform hyperparameter tuning using GridSearchCV.
    Get the best model and its parameters.
    """
    # drop nan values
    data_train = data_train.dropna()
    X_train = data_train.drop(columns=["date", target_column])
    y_train = data_train["close"]

    # Time Series Cross-Validation for train sets (train and validation)
    tscv = TimeSeriesSplit(n_splits=5)

    # Define models and hyperparameters for hyperparameter tuning
    model_name = "model"
    model = Ridge()
    param_grid = {"model__alpha": [0.05, 0.1, 0.2, 1.0]}
    print(f"\nTraining {model_name} model...")

    pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("model", model)]  # Handle NaNs in the dataset
    )

    # Perform Grid Search with TimeSeriesSplit if a parameter grid is provided
    if param_grid:
        search = GridSearchCV(
            pipeline, param_grid, cv=tscv, scoring="neg_mean_squared_error", error_score="raise"
        )
        # breakpoint()
        search.fit(X_train, y_train)
        best_model = search.best_estimator_  # average of valid datasets
        best_params = search.best_params_

    else:
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        best_params = None

    # Evaluate the model on the validation set
    # y_pred_val = best_model.predict(X_val)
    # val_mse = mean_squared_error(y_val, y_pred_val)
    # val_rmse = np.sqrt(val_mse)
    # val_mae = mean_absolute_error(y_val, y_pred_val)
    # val_r2 = r2_score(y_val, y_pred_val)

    feature_importance_analysis(best_model, X_train, y_train)
    detect_bias(data_train)

    local_model_path = "artifacts/models/best_linear_regression_model.joblib"
    joblib.dump(best_model, local_model_path)

    # folder = "artifacts"
    # if not os.path.exists(folder):
    #     os.makedirs(folder)

    # output_path = f"{folder}/{model_name}.joblib"
    # gcs_model_path = f"model_checkpoints/{model_name}.joblib"
    # save_and_upload_model(model=best_model, local_model_path=output_path, gcs_model_path=gcs_model_path)
    # return best_model, best_params

    return best_model, best_params


# Function to perform time series regression pipeline with model training and evaluation
def predict_linear_regression(data_test, target_column="close"):

    # load best model
    best_model = joblib.load("artifacts/models/best_linear_regression_model.joblib")
    print(f"best model: {best_model}")

    # data_test is scaled data
    date_test = data_test["date"]
    X_test = data_test.drop(columns=["date", target_column])
    # breakpoint()
    y_test = data_test[target_column]

    pred_metrics = {}

    # Predict using best model
    linear_regression_best_model = {}
    model_name = "Ridge Regression"
    y_pred = best_model.predict(X_test)

    # Evaluate the best model on the test set (hold-out dataset)
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    pred_metrics[model_name] = {
        "test_MSE": test_mse,
        "test_RMSE": test_rmse,
        "test_MAE": test_mae,
        "test_R2": test_r2,
    }

    # Store the trained model for plotting later
    linear_regression_best_model[model_name] = best_model

    # Plot Actual vs Predicted for each model
    plot_actual_vs_predicted_interactive(model_name, date_test, y_pred, y_test)

    return pred_metrics, linear_regression_best_model, y_test, y_pred


if __name__ == "__main__":
    scaled_train = pd.read_csv("pipeline/airflow/dags/data/scaled_data_train.csv")
    scaled_test = pd.read_csv("pipeline/airflow/dags/data/scaled_data_test.csv")
    # best_model, best_params = train_linear_regression(scaled_train, target_column="close")
    pred_metrics, linear_regression_best_model, y_test, y_pred = predict_linear_regression(
        scaled_test, target_column="close"
    )
    # print(f"y_pred: {y_pred}")
    # print(f"len_y_pred: {len(y_pred)}")
    # print(f"y_test: {y_test}")
    # print(f"len_y_test: {len(y_test)}")
    # print(f"best_params: {best_params}")
