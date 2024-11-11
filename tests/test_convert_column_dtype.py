import pytest
import pandas as pd
import numpy as np
import sys
import os


src_path = os.path.abspath("pipeline/airflow/dags/src")
sys.path.append(src_path)
from convert_column_dtype import convert_type_of_columns


def test_convert_type_of_columns():
    # Create a sample DataFrame with mixed types
    sample_data = pd.DataFrame(
        {
            "date": ["2021-07-13", "2021-07-14", "2021-07-15"],
            "float_col": ["1.1", "5.3", "6.7"],
            "object_col": ["a", "b", "c"],
        }
    )

    result = convert_type_of_columns(sample_data)

    # Check if date column is converted to datetime
    assert pd.api.types.is_datetime64_any_dtype(result["date"])

    # Check if numeric columns are converted to float
    assert result["float_col"].dtype == "float64"
    assert result["object_col"].dtype == "float64"

    # Check if all values are preserved
    assert result["float_col"].tolist() == [1.1, 5.3, 6.7]
