import pytest
import pandas as pd
import numpy as np
import sys
import os


# Add the src directory to the Python path
src_path = os.path.abspath("pipeline/airflow/dags/src")
sys.path.append(src_path)
from feature_interactions import add_feature_interactions


def test_add_feature_interactions():
    # Create a sample DataFrame
    data = pd.DataFrame(
        {
            "date": pd.date_range(start="2021-01-01", periods=10),
            "close": np.random.rand(10) * 100 + 100,
            "SP500": np.random.rand(10) * 1000 + 3000,
            "VIXCLS": np.random.rand(10) * 20 + 10,
        }
    )

    result = add_feature_interactions(data)

    # Check if new columns are added
    expected_new_columns = [
        "close_SP500_product",
        "close_SP500_ratio",
        "close_VIXCLS_product",
        "close_VIXCLS_ratio",
        "SP500_VIXCLS_product",
        "SP500_VIXCLS_ratio",
    ]
    for col in expected_new_columns:
        assert col in result.columns

    # Check if interaction features are correctly calculated
    assert np.isclose(result["close_SP500_product"].iloc[0], data["close"].iloc[0] * data["SP500"].iloc[0])
    assert np.isclose(
        result["close_SP500_ratio"].iloc[0], data["close"].iloc[0] / (data["SP500"].iloc[0] + 1e-5)
    )
    assert np.isclose(result["close_VIXCLS_product"].iloc[0], data["close"].iloc[0] * data["VIXCLS"].iloc[0])
    assert np.isclose(
        result["close_VIXCLS_ratio"].iloc[0], data["close"].iloc[0] / (data["VIXCLS"].iloc[0] + 1e-5)
    )
    assert np.isclose(result["SP500_VIXCLS_product"].iloc[0], data["SP500"].iloc[0] * data["VIXCLS"].iloc[0])
    assert np.isclose(
        result["SP500_VIXCLS_ratio"].iloc[0], data["SP500"].iloc[0] / (data["VIXCLS"].iloc[0] + 1e-5)
    )


def test_add_feature_interactions_missing_columns():
    incomplete_df = pd.DataFrame(
        {
            "date": pd.date_range(start="2021-01-01", periods=5),
            "close": [100, 101, 102, 103, 104],
            "SP500": [3000, 3001, 3002, 3003, 3004],
        }
    )
    with pytest.raises(KeyError):
        add_feature_interactions(incomplete_df)


def test_add_feature_interactions_single_row():
    single_row_df = pd.DataFrame(
        {
            "date": ["2021-01-01"],
            "close": [100],
            "SP500": [3000],
            "VIXCLS": [20],
        }
    )
    result = add_feature_interactions(single_row_df)
    assert not result.empty
    assert "close_SP500_product" in result.columns
    assert "close_SP500_ratio" in result.columns
    assert "close_VIXCLS_product" in result.columns
    assert "close_VIXCLS_ratio" in result.columns
    assert "SP500_VIXCLS_product" in result.columns
    assert "SP500_VIXCLS_ratio" in result.columns
