import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath("pipeline/airflow"))

from dags.src.scaler import scaler


@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2021-01-01", periods=100)
    data = pd.DataFrame(
        {
            "date": dates,
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100) * 10 + 5,
            "feature3": np.random.randn(100) * 100 - 50,
        }
    )
    return data


def test_scaler_training_data(sample_data):
    scaled_data = scaler(sample_data)

    # Check if scaled_data is a DataFrame
    assert isinstance(scaled_data, pd.DataFrame)

    # Check if date column is preserved
    assert "date" in scaled_data.columns
    assert (scaled_data["date"] == sample_data["date"]).all()

    # # Check if other columns are scaled
    # for col in ["feature1", "feature2", "feature3"]:
    #     assert np.isclose(scaled_data[col].mean(), 0, atol=1e-7)
    #     assert np.isclose(scaled_data[col].var(), 1, atol=1e-7)

    # # Check if mean and variance are returned correctly
    # assert isinstance(mean, np.ndarray)
    # assert isinstance(variance, np.ndarray)
    # assert len(mean) == len(variance) == 3  # number of features


def test_scaler_empty_df():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        scaler(empty_df)


def test_scaler_non_numeric():
    non_numeric_df = pd.DataFrame(
        {"date": pd.date_range(start="2021-01-01", periods=5), "feature1": ["a", "b", "c", "d", "e"]}
    )
    with pytest.raises(ValueError):
        scaler(non_numeric_df)


if __name__ == "__main__":
    pytest.main([__file__])
