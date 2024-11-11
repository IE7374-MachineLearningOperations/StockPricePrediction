import pytest
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.abspath("pipeline/airflow/dags/src"))

from download_data import (
    get_yfinance_data,
    get_fama_french_data,
    get_ads_index_data,
    get_sp500_data,
    get_fred_data,
    merge_data,
)


# Checking data is correctly downloaded for each dataset
def test_get_yfinance_data():
    data = get_yfinance_data("GOOGL")
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ["date", "open", "high", "low", "close", "volume"])


def test_get_fama_french_data():
    data = get_fama_french_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "date" in data.columns


def test_get_ads_index_data():
    data = get_ads_index_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ["date", "ads_index"])


def test_get_sp500_data():
    data = get_sp500_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ["date", "SP500"])


def test_get_fred_data():
    data = get_fred_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "date" in data.columns


def test_merge_data():
    data = merge_data("GOOGL")
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "date" in data.columns
    assert all(
        col in data.columns for col in ["open", "high", "low", "close", "volume", "SP500", "ads_index"]
    )


@pytest.mark.parametrize("ticker", ["GOOGL", "AAPL", "MSFT"])
def test_merge_data_multiple_tickers(ticker):
    data = merge_data(ticker)
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "date" in data.columns
    assert all(col in data.columns for col in ["open", "high", "low", "close", "volume"])


# Date should be unique and increasing
def test_merge_data_date_sorting():
    data = merge_data("GOOGL")
    assert data["date"].is_monotonic_increasing


def test_merge_data_no_unnamed_columns():
    data = merge_data("GOOGL")
    assert not any("Unnamed" in col for col in data.columns)


# Checking data is correctly downloaded for each dataset
def test_get_yfinance_data():
    data = get_yfinance_data("GOOGL")
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ["date", "open", "high", "low", "close", "volume"])


def test_get_fama_french_data():
    data = get_fama_french_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "date" in data.columns


def test_get_ads_index_data():
    data = get_ads_index_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ["date", "ads_index"])


def test_get_sp500_data():
    data = get_sp500_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert all(col in data.columns for col in ["date", "SP500"])


def test_get_fred_data():
    data = get_fred_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "date" in data.columns


def test_merge_data():
    data = merge_data("GOOGL")
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "date" in data.columns
    assert all(
        col in data.columns for col in ["open", "high", "low", "close", "volume", "SP500", "ads_index"]
    )


@pytest.mark.parametrize("ticker", ["GOOGL", "AAPL", "MSFT"])
def test_merge_data_multiple_tickers(ticker):
    data = merge_data(ticker)
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "date" in data.columns
    assert all(col in data.columns for col in ["open", "high", "low", "close", "volume"])


# Date should be unique and increasing
def test_merge_data_date_sorting():
    data = merge_data("GOOGL")
    assert data["date"].is_monotonic_increasing


def test_merge_data_no_unnamed_columns():
    data = merge_data("GOOGL")
    assert not any("Unnamed" in col for col in data.columns)
