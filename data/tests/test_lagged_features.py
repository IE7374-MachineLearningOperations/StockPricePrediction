import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('pipeline/airflow'))

from dags.src.lagged_features import add_lagged_features

def test_add_lagged_features():
    # Set a seed for reproducibility
    np.random.seed(0)
    
    # Create a sample DataFrame
    dates = pd.date_range(start='2021-01-01', periods=20)
    sample_data = pd.DataFrame({
        'date': dates,
        'close': np.random.rand(20) * 100 + 100,
        'open': np.random.rand(20) * 100 + 100,
        'high': np.random.rand(20) * 100 + 110,
        'low': np.random.rand(20) * 100 + 90
    })

    result = add_lagged_features(sample_data)

    # Check if new columns are added
    expected_new_columns = [
        'close_lag1', 'close_lag3', 'close_lag5',
        'open_lag1', 'open_lag3', 'open_lag5',
        'high_lag1', 'high_lag3', 'high_lag5',
        'low_lag1', 'low_lag3', 'low_lag5',
        'close_ma5', 'close_ma10', 'close_vol5', 'close_vol10'
    ]
    for col in expected_new_columns:
        assert col in result.columns

    # Check if lagged features are correctly calculated
    assert np.isclose(result['close_lag1'].iloc[10], sample_data['close'].iloc[9])
    assert np.isclose(result['open_lag3'].iloc[10], sample_data['open'].iloc[7])
    assert np.isclose(result['high_lag5'].iloc[10], sample_data['high'].iloc[5])

    # Check if rolling statistics are correctly calculated
    assert np.isclose(result['close_ma5'].iloc[10], sample_data['close'].iloc[6:11].mean())
    assert np.isclose(result['close_vol10'].iloc[15], sample_data['close'].iloc[6:16].std())

    # Check if NaN values are dropped
    assert not result.isnull().any().any()

    # Check if the number of rows is correct (should be 10 less due to 10-day window)
    assert len(result) == len(sample_data) - 10

def test_add_lagged_features_empty_df():
    empty_df = pd.DataFrame()
    result = add_lagged_features(empty_df)
    assert result.empty

def test_add_lagged_features_missing_columns():
    incomplete_df = pd.DataFrame({
        'date': pd.date_range(start='2021-01-01', periods=5),
        'close': [100, 101, 102, 103, 104]
    })
    with pytest.raises(KeyError):
        add_lagged_features(incomplete_df)

if __name__ == "__main__":
    pytest.main([__file__])
