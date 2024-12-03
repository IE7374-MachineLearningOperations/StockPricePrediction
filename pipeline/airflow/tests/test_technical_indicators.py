import pytest
import pandas as pd
import numpy as np
import sys
import os


# Add the src directory to the Python path
src_path = os.path.abspath("pipeline/airflow/dags/src")
sys.path.append(src_path)
from technical_indicators import add_technical_indicators

## fix random seed for reproducibility
np.random.seed(2024)


# @pytest.fixture
def sample_data():
    dates = pd.date_range(start="2021-01-01", periods=100)
    data = pd.DataFrame(
        {
            "date": dates,
            "close": np.random.rand(100) * 100 + 100,
            "open": np.random.rand(100) * 100 + 100,
            "high": np.random.rand(100) * 100 + 110,
            "low": np.random.rand(100) * 100 + 90,
        }
    )
    return data


def test_add_technical_indicators():
    data = sample_data()
    result = add_technical_indicators(data)

    # Check if new columns are added
    new_columns = ["RSI", "MACD", "MACD_signal", "MA20", "BB_upper", "BB_lower"]
    for col in new_columns:
        assert col in result.columns

    # Check RSI
    assert all(0 <= rsi <= 100 for rsi in result["RSI"])

    # Check MACD
    assert len(result["MACD"]) == len(result["MACD_signal"])

    # Check if NaN values are dropped
    assert not result.isnull().any().any()

    # Check if the number of rows is less than the original (due to NaN dropping)
    assert len(result) < len(data)


def test_add_technical_indicators_constant_price():
    # Create a DataFrame with constant price
    dates = pd.date_range(start="2021-01-01", periods=100)
    data = pd.DataFrame(
        {"date": dates, "close": [100] * 100, "open": [100] * 100, "high": [100] * 100, "low": [100] * 100}
    )

    result = add_technical_indicators(data)

    # Check MACD (should be close to 0 for constant price)
    assert all(np.isclose(macd, 0, atol=1e-5) for macd in result["MACD"])

    # Check Bollinger Bands (should be equal to price for constant price)
    assert all(np.isclose(result["BB_lower"], 100, atol=1e-5))
    assert all(np.isclose(result["BB_upper"], 100, atol=1e-5))


def test_add_technical_indicators_empty_df():
    empty_df = pd.DataFrame()
    with pytest.raises(KeyError):
        add_technical_indicators(empty_df)


def test_add_technical_indicators_missing_columns():
    incomplete_df = pd.DataFrame(
        {"date": pd.date_range(start="2021-01-01", periods=5), "close": [100, 101, 102, 103, 104]}
    )
    # This should run without error, as only 'close' is used
    result = add_technical_indicators(incomplete_df)
    assert "RSI" in result.columns


if __name__ == "__main__":
    pytest.main([__file__])
