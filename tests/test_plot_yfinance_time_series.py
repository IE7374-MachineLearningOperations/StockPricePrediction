import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pytest
from datetime import datetime, timedelta

# Add the src directory to the Python path
src_path = os.path.abspath("pipeline/airflow/dags/src")
sys.path.append(src_path)
from plot_yfinance_time_series import *


@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.random.rand(10) * 100 + 100,
            "high": np.random.rand(10) * 100 + 110,
            "low": np.random.rand(10) * 100 + 90,
            "close": np.random.rand(10) * 100 + 105,
            "volume": np.random.randint(1000, 10000, 10),
        }
    )


def test_plot_yfinance_time_series(sample_data):
    # Store the original savefig function
    original_savefig = plt.savefig

    # Replace plt.savefig with a dummy function
    plt.savefig = lambda *args, **kwargs: None

    try:
        # Call the function
        plot_yfinance_time_series(sample_data)

        # Check if the figure was created
        assert plt.gcf() is not None

        # Check if the correct number of subplots were created
        assert len(plt.gcf().axes) == 5  # open, high, low, close, volume

        # Check if the date was set as index
        assert sample_data.index.name == "date"

    finally:
        # Restore the original savefig function
        plt.savefig = original_savefig

        # Close all plots to free up memory
        plt.close("all")


def test_plot_yfinance_time_series_empty_data():
    empty_data = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    # Store the original savefig function
    original_savefig = plt.savefig

    # Replace plt.savefig with a dummy function
    plt.savefig = lambda *args, **kwargs: None

    try:
        # Call the function
        plot_yfinance_time_series(empty_data)

        # Check if the figure was created (even if empty)
        assert plt.gcf() is not None

    finally:
        # Restore the original savefig function
        plt.savefig = original_savefig

        # Close all plots to free up memory
        plt.close("all")
