import numpy as np
import pandas as pd
import sys
import os
import logging
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import joblib
import pickle
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler


# Function to plot Actual vs Predicted using Time Series Graph
def plot_actual_vs_predicted(models, X_test, y_test):
    plt.figure(figsize=(12, 8))

    # Plot actual values with date as x-axis
    plt.plot(y_test.index, y_test.values, label="Actual", color="black", linewidth=2, alpha=0.7)

    # Plot predicted values for each model using the same x-axis (dates)
    for model_name, model in models.items():
        y_pred_test = model.predict(X_test)
        plt.plot(y_test.index, y_pred_test, label=f"Predicted ({model_name})", alpha=0.7)

    plt.title("Actual vs Predicted Values")
    plt.xlabel("Date")
    plt.ylabel("Close Value")
    plt.legend()
    plt.grid(True)

    ax = plt.gca()

    # Set major ticks with fewer intervals based on the index length
    tick_frequency = len(y_test) // 10  # Shows one tick every 10% of the data
    ax.set_xticks(y_test.index[::tick_frequency])

    plt.xticks(rotation=90, size=8)
    plt.tight_layout()
    # plt.show()


def feature_importance_analysis(model, X_test, y_test):
    # Permutation feature importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

    # Sort features by importance
    sorted_idx = perm_importance.importances_mean.argsort()

    # Select top 10 features
    top_10_idx = sorted_idx[-10:]

    plt.figure(figsize=(10, 6))
    plt.barh(range(10), perm_importance.importances_mean[top_10_idx])
    plt.yticks(range(10), X_test.columns[top_10_idx])
    plt.title("Top 10 Features - Permutation Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    # plt.show()


def split_time_series_data(df, target_column="close", test_size=0.1, random_state=2024):
    # Sort the DataFrame by date
    df = df.sort_index()

    # Split features and target
    X = df.drop(columns=[target_column, "date"])
    y = df[target_column]

    # Split into train+val and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=2024
    )

    return X_train, X_test, y_train, y_test


# Function to train and evaluate models with hyperparameter tuning and time series split
def train_lr(
    model, data, target_column="close", test_size=0.1, param_grid=None, tscv=None, random_state=2024
):
    """
    Train a model on the given data and evaluate it using TimeSeriesSplit cross-validation.
    Optionally perform hyperparameter tuning using GridSearchCV.
    Get the best model and its parameters.
    """
    X_train, _, y_train, _ = split_time_series_data(data, target_column, test_size, random_state)

    pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("model", model)]  # Handle NaNs in the dataset
    )

    # Perform Grid Search with TimeSeriesSplit if a parameter grid is provided
    if param_grid:
        search = GridSearchCV(
            pipeline, param_grid, cv=tscv, scoring="neg_mean_squared_error", error_score="raise"
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_  # average of valid datasets
        best_params = search.best_params_

        # Hyperparameter Sensitivity Analysis
        results = pd.DataFrame(search.cv_results_)
        for param in param_grid:
            plt.figure(figsize=(10, 6))
            plt.plot(results[f"param_{param}"], -results["mean_test_score"])
            plt.xlabel(param)
            plt.ylabel("Mean Squared Error")
            plt.title(f"Hyperparameter Sensitivity: {param}")
            plt.xscale("log")
            # plt.show()

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
    # save best model locally
    # save_and_upload_model(best_model, "best_model.pkl")
    # best_model.save("best_model.pkl")
    local_model_path = "artifacts/models/best_linear_regression_model.pkl"
    joblib.dump(best_model, local_model_path)

    return best_model, best_params


# Function to perform time series regression pipeline with model training and evaluation
def predict_lr(data, target_column="close", test_size=0.1, random_state=2024):
    # with open("artifacts/models/best_linear_regression_model.pkl", "rb") as f:
    #     best_model = pickle.load(f)
    best_model = joblib.load("artifacts/models/best_linear_regression_model.pkl")
    print(f"best model: {best_model}")
    pred_metrics = {}

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_time_series_data(data, target_column, test_size)

    # Time Series Cross-Validation for train sets (train and validation)
    tscv = TimeSeriesSplit(n_splits=5)

    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    # y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))
    # y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))

    # Define models and hyperparameters for hyperparameter tuning
    lr_best_model = {}
    model_name = "Ridge Regression"
    model = Ridge()
    param_grid = {"model__alpha": [0.05, 0.1, 0.2, 1.0]}
    print(f"\nTraining {model_name} model...")
    # best_model, _ = train_lr(
    #     model, data, target_column="close", test_size=0.1, param_grid=param_grid, tscv=tscv, random_state=2024
    # )

    # Evaluate the best model on the test set (hold-out dataset)
    y_pred = best_model.predict(X_test)
    # y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    # y_test_unscaled = scaler.inverse_transform(y_test_scaled.reshape(-1, 1))

    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    # Evaluate the best model on the test set (hold-out dataset)
    # y_pred = best_model.predict(X_test)
    # test_mse = mean_squared_error(y_test, y_pred)
    # test_rmse = np.sqrt(test_mse)
    # test_mae = mean_absolute_error(y_test, y_pred)
    # test_r2 = r2_score(y_test, y_pred)

    # Store results in the results dictionary (including validation and test results)
    pred_metrics[model_name] = {
        "test_MSE": test_mse,
        "test_RMSE": test_rmse,
        "test_MAE": test_mae,
        "test_R2": test_r2,
    }

    # Store the trained model for plotting later
    lr_best_model[model_name] = best_model

    # Print results
    # for model_name, metrics in pred_metrics.items():
    #     print(
    #         f"\n{model_name}_Validation_MSE: {metrics['val_MSE']:.4f}, {model_name}_Validation_RMSE: {metrics['val_RMSE']:.4f}, "
    #         f"{model_name}_Validation_MAE: {metrics['val_MAE']:.4f}, {model_name}_Validation_R2: {metrics['val_R2']:.4f}"
    #     )
    #     print(
    #         f"{model_name}_Test_MSE: {metrics['test_MSE']:.4f}, {model_name}_Test_RMSE: {metrics['test_RMSE']:.4f}, "
    #         f"{model_name}_Test_MAE: {metrics['test_MAE']:.4f}, {model_name}_Test_R2: {metrics['test_R2']:.4f}"
    #     )

    # Plot Actual vs Predicted for each model
    plot_actual_vs_predicted(lr_best_model, X_test, y_test)

    feature_importance_analysis(best_model, X_test, y_test)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values, feature_names=X_train.columns, class_names=["close"], mode="regression"
    )

    # Choose a random instance to explain
    instance = X_test.iloc[np.random.randint(0, len(X_test))].values

    # Generate the explanation
    exp = explainer.explain_instance(instance, best_model.predict, num_features=10)

    # Visualize the explanation
    # exp.show_in_notebook(show_table=True)

    print(f"\nFeature Importances for {model_name}:")
    for feature, importance in exp.as_list():
        print(f"{feature}: {importance}")

    return pred_metrics, lr_best_model, y_test, y_pred


if __name__ == "__main__":
    data = pd.read_csv("pipeline/airflow/dags/data/final_dataset_for_modeling.csv")
    # model = Ridge()
    # best_model, best_params = train_lr(model, data, target_column="close", test_size=0.1, random_state=2024)
    pred_metrics, lr_best_model, y_test, y_pred = predict_lr(
        data, target_column="close", test_size=0.1, random_state=2024
    )
    print(y_pred)
