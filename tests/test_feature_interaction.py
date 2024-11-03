import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath('pipeline/airflow/dags/src'))

from feature_interactions import add_feature_interactions

def test_add_feature_interactions():
    # Create a sample DataFrame with necessary columns
    sample_data = pd.DataFrame({
        'date': pd.date_range(start='2021-01-01', periods=5),
        'close': [100, 101, 99, 102, 103],
        'SP500': [3000, 3010, 2990, 3020, 3030],
        'VIXCLS': [20, 21, 19, 22, 18]
    })

    result = add_feature_interactions(sample_data)

    # Check if new columns are added
    new_columns = ['close_SP500_product', 'close_SP500_ratio', 
                   'close_VIXCLS_product', 'close_VIXCLS_ratio',
                   'SP500_VIXCLS_product', 'SP500_VIXCLS_ratio']
    for col in new_columns:
        assert col in result.columns

    # Check if calculations are correct
    assert result['close_SP500_product'].iloc[0] == 100 * 3000
    assert result['close_SP500_ratio'].iloc[0] == 100 / 3000
    assert result['close_VIXCLS_product'].iloc[0] == 100 * 20
    assert result['close_VIXCLS_ratio'].iloc[0] == 100 / 20
    assert result['SP500_VIXCLS_product'].iloc[0] == 3000 * 20
    assert result['SP500_VIXCLS_ratio'].iloc[0] == 3000 / 20

    # Check if original columns are preserved
    assert 'close' in result.columns
    assert 'SP500' in result.columns
    assert 'VIXCLS' in result.columns

    # Check if the number of rows remains the same
    assert len(result) == len(sample_data)