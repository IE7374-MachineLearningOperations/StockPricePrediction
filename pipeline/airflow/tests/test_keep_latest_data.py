import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add the src directory to the Python path
src_path = os.path.abspath("pipeline/airflow/dags/src")
sys.path.append(src_path)
from keep_latest_data import keep_latest_data


def test_keep_latest_data():
    # Create a sample DataFrame with date column
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 12)  # 12 years of data
    sample_data = pd.DataFrame(
        {
            "date": pd.date_range(start=start_date, end=end_date, freq="D"),
            "value": range((end_date - start_date).days + 1),
        }
    )

    # Test keeping data for the last 5 years
    result_5_years = keep_latest_data(sample_data, 5)

    # Check if the result has the correct number of rows (approximately)
    assert abs(len(result_5_years) - 365 * 5) < 10

    # Check if all dates in the result are within the last 5 years
    five_years_ago = datetime.now() - timedelta(days=365 * 5) - timedelta(days=2)
    assert result_5_years["date"].min() >= five_years_ago

    # Test with num_years greater than the available data
    result_all = keep_latest_data(sample_data, 15)
    assert len(result_all) == len(sample_data)  # Should return all rows


def test_keep_latest_data_empty_df():
    empty_df = pd.DataFrame(columns=["date", "value"])
    result = keep_latest_data(empty_df, 5)
    assert result.empty


def test_keep_latest_data_single_day():
    single_day_df = pd.DataFrame({"date": [datetime.now()], "value": [1]})

    result = keep_latest_data(single_day_df, 5)
    assert len(result) == 1
