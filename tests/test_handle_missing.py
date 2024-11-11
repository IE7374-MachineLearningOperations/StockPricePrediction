import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path

src_path = os.path.abspath("pipeline/airflow/dags/src")
sys.path.append(src_path)
from handle_missing import fill_missing_values


def test_fill_missing_values_basic():
    # Create a sample DataFrame with missing values
    sample_data = pd.DataFrame({"low": [np.nan, np.nan, 3, np.nan, 5], "high": [np.nan, 2, np.nan, 4, 5]})

    result = fill_missing_values(sample_data)

    # Check if there are no missing values in the result
    assert result.isna().sum().sum() == 0

    # Check if the backward fill and forward fill worked correctly

    assert result["low"].tolist() == [3, 3, 3, 5, 5]
    assert result["high"].tolist() == [2, 2, 4, 4, 5]


def test_fill_missing_values_all_nan_column():
    nan_data = pd.DataFrame({"A": [1, 2, 3], "low": [2, np.nan, np.nan], "high": [np.nan, np.nan, np.nan]})

    nan_result = fill_missing_values(nan_data)

    # Check if there are no missing values in the result
    assert nan_result.isna().sum().sum() == 0

    # Check if the all-NaN column is dropped
    assert "high" not in nan_result.columns

    # Check if the partially NaN column is filled correctly
    assert nan_result["low"].tolist() == [2, 2, 2]


def test_fill_missing_values_empty_dataframe():
    empty_df = pd.DataFrame()
    result = fill_missing_values(empty_df)
    assert result.empty


def test_fill_missing_values_no_missing_data():
    complete_data = pd.DataFrame({"low": [1, 2, 3], "high": [4, 5, 6]})

    result = fill_missing_values(complete_data)
    pd.testing.assert_frame_equal(result, complete_data)
