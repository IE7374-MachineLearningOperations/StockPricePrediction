import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath("pipeline/airflow"))
from dags.src.pca import apply_pca, visualize_pca_components


@pytest.fixture
def sample_data():
    np.random.seed(42)
    dates = pd.date_range(start="2021-01-01", periods=100)
    data = pd.DataFrame(
        {
            "date": dates,
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "feature4": np.random.randn(100),
            "feature5": np.random.randn(100),
        }
    )
    return data


def test_apply_pca(sample_data):
    reduced_data, explained_variance, n_components = apply_pca(sample_data, variance_threshold=0.95)

    # Check if reduced_data is a numpy array
    assert isinstance(reduced_data, np.ndarray)

    # Check if explained_variance is a numpy array
    assert isinstance(explained_variance, np.ndarray)

    # Check if n_components is an integer
    assert isinstance(n_components, int)

    # Check if the shape of reduced_data is correct
    assert reduced_data.shape[0] == len(sample_data)
    assert reduced_data.shape[1] <= n_components

    # Check if the explained variance sums to approximately 1
    assert np.isclose(np.sum(explained_variance), 1, atol=1e-5)

    # Check if the variance explained is at least the threshold
    assert np.sum(explained_variance[:n_components]) >= 0.95


def test_visualize_pca_components(sample_data, monkeypatch):
    # Mock plt.show to avoid displaying the plot during testing
    monkeypatch.setattr(plt, "show", lambda: None)

    # Mock plt.savefig to avoid saving the plot during testing
    monkeypatch.setattr(plt, "savefig", lambda x: None)

    # Call the function
    visualize_pca_components(sample_data, variance_threshold=0.95)

    # Check if the plot was created (plt.gcf() gets the current figure)
    assert plt.gcf() is not None

    # Check if the plot has the correct labels and title
    ax = plt.gca()
    assert ax.get_xlabel() == "Principal Component 1"
    assert ax.get_ylabel() == "Principal Component 2"
    assert ax.get_title() == "PCA Components"


def test_apply_pca_empty_df():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        apply_pca(empty_df)


def test_apply_pca_non_numeric():
    non_numeric_df = pd.DataFrame(
        {"date": pd.date_range(start="2021-01-01", periods=5), "feature1": ["a", "b", "c", "d", "e"]}
    )
    with pytest.raises(ValueError):
        apply_pca(non_numeric_df)


if __name__ == "__main__":
    pytest.main([__file__])
