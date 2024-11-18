import pytest
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.append(os.path.abspath("pipeline/airflow/dags/src"))
from correlation import plot_correlation_matrix, removing_correlated_variables


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [5, 4, 3, 2, 1],
            "C": [1, 1, 1, 1, 1],
            "D": [1, 2, 3, 4, 5],
            "E": [2, 4, 6, 8, 10],
            "open": [10, 20, 30, 40, 50],
            "high": [11, 21, 31, 41, 51],
            "low": [9, 19, 29, 39, 49],
            "close": [10.5, 20.5, 30.5, 40.5, 50.5],
            "volume": [1000, 2000, 3000, 4000, 5000],
            "SP500": [100, 101, 102, 103, 104],
        }
    )


def test_plot_correlation_matrix(sample_data):
    # Store the original savefig function
    original_savefig = plt.savefig

    # Replace plt.savefig with a dummy function
    plt.savefig = lambda *args, **kwargs: None

    try:
        plot_correlation_matrix(sample_data)
        # If no exception is raised, we assume the function ran successfully
    finally:
        # Restore the original savefig function
        plt.savefig = original_savefig


def test_removing_correlated_variables(sample_data):
    # Store the original savefig function
    original_savefig = plt.savefig

    # Replace plt.savefig with a dummy function
    plt.savefig = lambda *args, **kwargs: None

    try:
        result = removing_correlated_variables(sample_data)

        # Check if highly correlated columns are removed
        assert "C" not in result.columns  # C is constant, should be removed
        assert "E" not in result.columns  # E is perfectly correlated with D, should be removed

        # Check if important columns are retained
        assert all(col in result.columns for col in ["open", "high", "low", "close", "volume", "SP500"])
    finally:
        # Restore the original savefig function
        plt.savefig = original_savefig


def test_removing_correlated_variables_with_nan():
    sample_data_with_nan = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [5, 4, 3, 2, 1],
            "C": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "open": [10, 20, 30, 40, 50],
            "high": [11, 21, 31, 41, 51],
            "low": [9, 19, 29, 39, 49],
            "close": [10.5, 20.5, 30.5, 40.5, 50.5],
            "volume": [1000, 2000, 3000, 4000, 5000],
            "SP500": [100, 101, 102, 103, 104],
        }
    )

    # Store the original savefig function
    original_savefig = plt.savefig

    # Replace plt.savefig with a dummy function
    plt.savefig = lambda *args, **kwargs: None

    try:
        result = removing_correlated_variables(sample_data_with_nan)
        assert "C" not in result.columns  # C is all NaN, should be removed
    finally:
        # Restore the original savefig function
        plt.savefig = original_savefig


def test_removing_correlated_variables_no_removal():
    # Create a dataset where no columns should be removed
    data = pd.DataFrame(
        {
            "open": [10, 20, 30, 40, 50],
            "high": [11, 21, 31, 41, 51],
            "low": [9, 19, 29, 39, 49],
            "close": [10.5, 20.5, 30.5, 40.5, 50.5],
            "volume": [1000, 2000, 3000, 4000, 5000],
            "SP500": [100, 101, 102, 103, 104],
        }
    )

    # Store the original savefig function
    original_savefig = plt.savefig

    # Replace plt.savefig with a dummy function
    plt.savefig = lambda *args, **kwargs: None

    try:
        result = removing_correlated_variables(data)
        assert set(result.columns) == set(data.columns)
    finally:
        # Restore the original savefig function
        plt.savefig = original_savefig
