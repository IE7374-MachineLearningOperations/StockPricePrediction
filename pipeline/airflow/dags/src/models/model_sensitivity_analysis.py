import numpy as np
import pandas as pd
import sys
import os
import logging
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
from dags.src.models.model_utils import save_and_upload_model, upload_artifact


# Define the data preparation function
def prepare_data(data, test_size=0.2):
    X = data.drop(["close", "date"], axis=1)
    y = data["close"]

    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


# Define model training and evaluation function with hyperparameter tuning and imputation
def train_evaluate_model(model, X_train, y_train, tscv, param_grid=None):
    pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("model", model)]  # Handle NaNs in the dataset
    )

    if param_grid:
        search = GridSearchCV(pipeline, param_grid, cv=tscv, scoring="neg_mean_squared_error")
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        # print(f"Best Parameters: {best_params}")
        logging.info(f"Best Parameters: {best_params}")

        # Hyperparameter sensitivity analysis
        results = pd.DataFrame(search.cv_results_)
        for param in param_grid:
            plt.figure(figsize=(10, 6))
            plt.plot(results[f"param_{param}"], results["mean_test_score"])
            plt.xlabel(param)
            plt.ylabel("Mean test score")
            plt.title(f"Hyperparameter Sensitivity: {param}")
            # plt.show()
            # Ensure the assets directory exists
            if not os.path.exists("artifacts"):
                os.makedirs("artifacts")
                logging.info(f"Created directory: artifacts")

            fig_path = f"artifacts/Linear Regression - Hyperparameter Sensitivity: {param}.png"
            gcs_model_path = f"Data/pipeline/airflow/{fig_path}"
            plt.savefig(fig_path)
            upload_artifact(fig_path, gcs_model_path)
            # plt.savefig(f"artifacts/Linear Regression - Hyperparameter Sensitivity: {param}.png")
            logging.info(f"Linear Regression - Hyperparameter Sensitivity: {param}.png saved to artifacts")

    else:
        pipeline.fit(X_train, y_train)
        best_model = pipeline

    # Cross-validation for evaluation metrics
    mse_scores, mae_scores = [], []
    for train_idx, val_idx in tscv.split(X_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        best_model.fit(X_cv_train, y_cv_train)
        y_pred = best_model.predict(X_cv_val)

        mse_scores.append(mean_squared_error(y_cv_val, y_pred))
        mae_scores.append(mean_absolute_error(y_cv_val, y_pred))

    results = {"mean_mse": np.mean(mse_scores), "mean_mae": np.mean(mae_scores)}
    # print("Cross-Validation MSE:", results["mean_mse"])
    # print("Cross-Validation MAE:", results["mean_mae"])

    logging.info(f"Cross-Validation MSE: {results['mean_mse']}")
    logging.info(f"Cross-Validation MAE: {results['mean_mae']}")

    # Feature importance using SHAP
    logging.info("Running SHAP")
    explainer = shap.LinearExplainer(best_model["model"], X_train[:200])
    shap_values = explainer.shap_values(X_train[:200])

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.title(f"Feature Importance for {type(model).__name__}")
    plt.tight_layout()

    # Ensure the assets directory exists
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
        logging.info(f"Created directory: artifacts")

    fig_path = f"artifacts/Feature Importance for {type(model).__name__}.png"
    gcs_model_path = f"Data/pipeline/airflow/{fig_path}"
    plt.savefig(fig_path)
    upload_artifact(fig_path, gcs_model_path)
    logging.info(
        f"Feature Importance for {type(model).__name__} saved to artifacts/Feature Importance for {type(model).__name__}.png"
    )

    return best_model, results


# Main function for the full pipeline
def time_series_regression_pipeline(data, test_size=0.2):
    logging.info("Starting time series regression pipeline")
    X_train, X_test, y_train, y_test = prepare_data(data, test_size)

    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        # "Ridge": (Ridge(), {"model__alpha": [0.1, 0.5]}),
        # "Lasso": (Lasso(), {"model__alpha": [0.1, 0.5]}),
        "ElasticNet": (
            ElasticNet(),
            {"model__alpha": [0.1, 0.5], "model__l1_ratio": [0, 0.1, 0.5]},
        ),
    }

    final_models = {}
    logging.info("Training models...")
    for model_name, (model, param_grid) in models.items():
        logging.info(f"Training {model_name} model...")
        best_model, results = train_evaluate_model(model, X_train, y_train, tscv, param_grid)
        final_models[model_name] = best_model

        # Perform SHAP analysis on test set
        explainer = shap.LinearExplainer(best_model["model"], X_test)
        shap_values = explainer.shap_values(X_test)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title(f"Feature Importance for {model_name} on Test Set")
        plt.tight_layout()

        # Ensure the assets directory exists
        if not os.path.exists("artifacts"):
            os.makedirs("artifacts")
            logging.info(f"Created directory: artifacts")

        fig_path = f"artifacts/Feature Importance for {model_name} on Test Set.png"
        gcs_model_path = f"Data/pipeline/airflow/{fig_path}"
        plt.savefig(fig_path)
        upload_artifact(fig_path, gcs_model_path)
        logging.info(
            f"PCA components plot saved to artifacts/Feature Importance for {model_name} on Test Set.png"
        )

    # Evaluate final models on test set
    folder = "artifacts"
    if not os.path.exists(folder):
        os.makedirs(folder)

    logging.info("\nEvaluating final models on test set")
    for model_name, model in final_models.items():
        y_pred_test = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        logging.info(f"{model_name} Test MSE: {test_mse}, Test MAE: {test_mae}")

        logging.info(f"Uploading the best model '{model_name}' to Google Cloud Storage")
        output_path = f"{folder}/{model_name}.pkl"
        gcs_model_path = f"model_checkpoints/{model_name}.pkl"
        save_and_upload_model(model=model, local_model_path=output_path, gcs_model_path=gcs_model_path)

    return final_models, results


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
    final_models = time_series_regression_pipeline(scaled_data)
    print(final_models)
