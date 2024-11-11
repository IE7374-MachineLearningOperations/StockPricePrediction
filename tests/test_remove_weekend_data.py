import pytest
import pandas as pd
import numpy as np
import sys
import os


src_path = os.path.abspath("pipeline/airflow/dags/src")
sys.path.append(src_path)
from remove_weekend_data import remove_weekends


def test_remove_weekends():
    # Create a sample DataFrame with weekend dates
    dates = pd.date_range(
        start="2024-01-01", periods=10, freq="D"
    )  # 10 consecutive days starting from January 1, 2024
    sample_data = pd.DataFrame({"date": dates, "value": range(10)})

    # Remove weekends from the DataFrame
    result = remove_weekends(sample_data)

    # Check if the length of result is correct (Should remove Saturday and Sunday)
    assert len(result) == 8  # 8 weekdays in the range

    # Check if weekends are indeed removed

    assert all(result["date"].dt.weekday < 5)  # Check if all weekdays are less than 5 (Monday=0, Sunday=6)


def test_remove_weekends_no_weekends():
    # Create a sample DataFrame with only weekdays
    dates = pd.date_range(start="2021-01-04", periods=5, freq="D")  # 5 weekdays
    sample_data = pd.DataFrame({"date": dates, "value": range(5)})

    result = remove_weekends(sample_data)

    # Check if all data is preserved
    assert len(result) == 5

    assert all(result["date"].dt.weekday < 5)


def test_remove_weekends_all_weekends():
    # Create a sample DataFrame with only weekends

    dates = pd.to_datetime(["2021-01-02", "2021-01-03", "2021-01-09", "2021-01-10"])
    sample_data = pd.DataFrame({"date": dates, "value": range(4)})

    result = remove_weekends(sample_data)

    # Check if all data is removed
    assert result.empty
