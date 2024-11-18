import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import resample
from fairlearn.metrics import MetricFrame
import logging
import sys
import os

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
# from dags.src.correlation import removing_correlated_variables
# from dags.src.lagged_features import add_lagged_features
# from dags.src.feature_interactions import add_feature_interactions
# from dags.src.technical_indicators import add_technical_indicators
# from dags.src.scaler import scaler
# from dags.src.upload_blob import upload_blob
# from dags.src.models.model_utils import save_and_upload_model, upload_artifact


def detect_bias_1(data, model=None):

    # Define features and target
    features = [col for col in data.columns if col != "close" and col != "date"]
    target = "close"

    # Split data into training and test sets
    test_size = 0.2  # Adjust test size as needed
    X_train, X_test, y_train, y_test = train_test_split(
        data[features], data[target], test_size=test_size, shuffle=False
    )

    # Add the target column to the training set for resampling
    train_data = X_train.copy()
    train_data["close"] = y_train  # Add the target to the training features for resampling

    # Define the slice condition based on VIX values
    vix_median = pd.to_numeric(train_data["VIXCLS"], errors="coerce").median()
    high_vix_data = train_data[train_data["VIXCLS"] > vix_median]
    low_vix_data = train_data[train_data["VIXCLS"] <= vix_median]

    # Upsample high-error slice data
    high_vix_upsampled = resample(high_vix_data, replace=True, n_samples=len(low_vix_data), random_state=42)
    balanced_train_data = pd.concat([high_vix_upsampled, low_vix_data])

    # Separate features and target after balancing
    X_train_balanced = balanced_train_data.drop(columns="close")  # Drop the target from balanced data
    y_train_balanced = balanced_train_data["close"]  # Get the target values

    # Train the ElasticNet model on the balanced data
    model = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model.fit(X_train_balanced, y_train_balanced)

    # Predict and evaluate on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    logging.info(f"ElasticNet Test MSE with Resampling: {mse}")
    logging.info(f"ElasticNet Test MAE with Resampling: {mae}")


def detect_bias_2(data):

    features = [col for col in data.columns if col != "close" and col != "date"]
    target = "close"
    test_size = 0.2  # Adjust test size as needed

    X_train, X_test, y_train, y_test = train_test_split(
        data[features], data[target], test_size=test_size, shuffle=False
    )
    model = ElasticNet(alpha=1.0, l1_ratio=0.5)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    # Define thresholds for slicing (adjust thresholds if needed)
    sp500_median = data["SP500"].median()
    vix_median = data["VIXCLS"].median()

    # Define slice conditions using SP500 and VIXCLS quantiles
    sensitive_features = pd.Series(index=X_test.index, dtype="object")
    sensitive_features[X_test["SP500"] > sp500_median] = "high_sp500"
    sensitive_features[X_test["SP500"] <= sp500_median] = "low_sp500"
    sensitive_features[X_test["VIXCLS"] > vix_median] = "high_vix"
    sensitive_features[X_test["VIXCLS"] <= vix_median] = "low_vix"

    # Custom functions for MSE and MAE
    def mse_func(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def mae_func(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    # Calculate metrics across slices using MetricFrame
    mse_mae_metrics = MetricFrame(
        metrics={"MSE": mse_func, "MAE": mae_func},
        y_true=y_test,
        y_pred=y_pred_test,
        sensitive_features=sensitive_features,
    )

    # Display results for each slice
    logging.info("\nMetrics by Slice:")
    logging.info(f"{mse_mae_metrics.by_group}")

    # Convert MSE and MAE columns to numeric, in case they are not already
    mse_mae_metrics.by_group["MSE"] = pd.to_numeric(mse_mae_metrics.by_group["MSE"], errors="coerce")
    mse_mae_metrics.by_group["MAE"] = pd.to_numeric(mse_mae_metrics.by_group["MAE"], errors="coerce")

    # Calculate and display MSE and MAE disparities across slices
    mse_disparity = mse_mae_metrics.difference(method="between_groups")["MSE"]
    mae_disparity = mse_mae_metrics.difference(method="between_groups")["MAE"]
    logging.info(f"\nOverall MSE Disparity: {mse_disparity}")
    logging.info(f"Overall MAE Disparity: {mae_disparity}")

    # Identify and report on slices with the highest and lowest MSE for potential bias investigation
    highest_mse_slice = mse_mae_metrics.by_group["MSE"].idxmax()
    lowest_mse_slice = mse_mae_metrics.by_group["MSE"].idxmin()

    logging.info(
        f"\nSlice with highest MSE: {highest_mse_slice} - MSE: {mse_mae_metrics.by_group['MSE'][highest_mse_slice]}"
    )
    logging.info(
        f"Slice with lowest MSE: {lowest_mse_slice} - MSE: {mse_mae_metrics.by_group['MSE'][lowest_mse_slice]}"
    )


def detect_bias(data):
    detect_bias_1(data)
    detect_bias_2(data)
    logging.info("-----Done detecting bias !!! ------")


if __name__ == "__main__":
    # data = pd.read_csv(
    #     "/Users/vy/my_work/NEU_course/MLOps/StockPricePrediction/pipeline/airflow/dags/data/merged_original_dataset.csv"
    # )
    # ticker_symbol = "GOOGL"
    # data = merge_data(ticker_symbol)
    # data = convert_type_of_columns(data)
    # filtered_data = keep_latest_data(data, 10)
    # removed_weekend_data = remove_weekends(filtered_data)
    # filled_data = fill_missing_values(removed_weekend_data)
    # removed_correlated_data = removing_correlated_variables(filled_data)
    # lagged_data = add_lagged_features(removed_correlated_data)
    # feature_interactions_data = add_feature_interactions(lagged_data)
    # technical_indicators_data = add_technical_indicators(feature_interactions_data)
    # scaled_data = scaler(technical_indicators_data)
    # data = scaled_data
    # detect_bias(data)
    pass
