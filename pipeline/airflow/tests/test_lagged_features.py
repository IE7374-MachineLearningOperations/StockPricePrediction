import pytest
import pandas as pd
import numpy as np
import sys
import os


# Add the src directory to the Python path
src_path = os.path.abspath("pipeline/airflow/dags/src")
sys.path.append(src_path)
from lagged_features import add_lagged_features


def test_add_lagged_features():
    # Set a seed for reproducibility
    np.random.seed(0)

    # Create a sample DataFrame
    dates = pd.date_range(start="2021-01-01", periods=20)
    sample_data = pd.DataFrame(
        {
            "date": dates,
            "close": np.random.rand(20) * 100 + 100,
            "open": np.random.rand(20) * 100 + 100,
            "high": np.random.rand(20) * 100 + 110,
            "low": np.random.rand(20) * 100 + 90,
        }
    )

    result = add_lagged_features(sample_data)

    # Check if new columns are added
    expected_new_columns = [
        "close_lag1",
        "close_lag3",
        "close_lag5",
        "open_lag1",
        "open_lag3",
        "open_lag5",
        "high_lag1",
        "high_lag3",
        "high_lag5",
        "low_lag1",
        "low_lag3",
        "low_lag5",
        "close_ma5",
        "close_ma10",
        "close_vol5",
        "close_vol10",
    ]
    for col in expected_new_columns:
        assert col in result.columns

    # Check if NaN values are dropped
    assert not result.isnull().any().any()


def test_add_lagged_features_empty_df():
    empty_df = pd.DataFrame(columns=["date", "close", "open", "high", "low"])
    result = add_lagged_features(empty_df)
    assert result.empty


def test_add_lagged_features_missing_columns():
    incomplete_df = pd.DataFrame(
        {"date": pd.date_range(start="2021-01-01", periods=5), "close": [100, 101, 102, 103, 104]}
    )
    with pytest.raises(KeyError):
        add_lagged_features(incomplete_df)


def test_add_lagged_features_single_row():
    single_row_df = pd.DataFrame(
        {
            "date": ["2021-01-01"],
            "close": [100],
            "open": [100],
            "high": [110],
            "low": [90],
        }
    )
    result = add_lagged_features(single_row_df)
    assert result.empty


def test_add_lagged_features_all_nan():
    nan_df = pd.DataFrame(
        {
            "date": pd.date_range(start="2021-01-01", periods=5),
            "close": [np.nan] * 5,
            "open": [np.nan] * 5,
            "high": [np.nan] * 5,
            "low": [np.nan] * 5,
        }
    )
    result = add_lagged_features(nan_df)
    assert result.empty


def test_add_lagged_features_no_variation():
    no_variation_df = pd.DataFrame(
        {
            "date": pd.date_range(start="2021-01-01", periods=10),
            "close": [100] * 10,
            "open": [100] * 10,
            "high": [110] * 10,
            "low": [90] * 10,
        }
    )
    result = add_lagged_features(no_variation_df)
    assert (result["close_ma5"] == 100).all()
    assert (result["close_vol5"] == 0).all()


if __name__ == "__main__":
    pytest.main([__file__])
