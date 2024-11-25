import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
import glob
from datetime import datetime
import os
import logging
from google.cloud import storage
import io
import gcsfs

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


abs_path = os.path.abspath(__file__)
dir = os.path.dirname(abs_path)
dir = os.path.dirname(dir)
path = os.path.join(dir, "service_key_gcs.json")

if os.path.exists(path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
    storage_client = storage.Client()
else:
    storage_client = None
    logging.warning("------- Service key not found!")


def read_file(blob_name: str, bucket_name="stock_price_prediction_dataset") -> pd.DataFrame:
    """Write and read a blob from GCS using file-like IO"""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    with blob.open("r") as f:
        df = f.read()
        return df


def get_yfinance_data(ticker_symbol: str) -> pd.DataFrame:
    ticker = yf.Ticker(ticker_symbol)

    ## Fetch historical market data
    ## period = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    historical_data = ticker.history(period="max")
    historical_data.reset_index(inplace=True)
    historical_data["Date"] = historical_data["Date"].dt.date
    historical_data.columns = historical_data.columns.str.lower()
    historical_data.columns = historical_data.columns.str.replace(" ", "_")
    if historical_data.empty:
        logging.error(f"Historical data for {ticker_symbol} NOT fetched")
    logging.info(f"Historical data for {ticker_symbol} fetched successfully")

    return historical_data


def get_fama_french_data() -> pd.DataFrame:
    bucket_name = "stock_price_prediction_dataset"
    blob_name = "Data/pipeline/airflow/dags/data/fama_french.csv"
    if storage_client is not None:  ## connect to GCS
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        with blob.open("r") as f:
            ff = f.read()

        ff = pd.read_csv(io.StringIO(ff), sep=",")
    else:  ## local
        ff = pd.read_csv("pipeline/airflow/dags/data/fama_french.csv")

    ff["Date"] = pd.to_datetime(ff["Date"], format="%Y%m%d")
    all_cols = ff.columns
    new_cols = ["date"] + list(all_cols[1:])
    ff.columns = new_cols
    if ff.empty:
        logging.error("Fama French data was NOT loaded")
    logging.info("Fama French data was loaded successfully")

    return ff


def get_ads_index_data() -> pd.DataFrame:
    bucket_name = "stock_price_prediction_dataset"
    blob_name = "Data/pipeline/airflow/dags/data/ADS_index.csv"

    if storage_client is not None:  ## connect to GCS
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        with blob.open("r") as f:
            ads = f.read()
        ads = pd.read_csv(io.StringIO(ads), sep=",")
    else:  # local
        ads = pd.read_csv("pipeline/airflow/dags/data/ADS_index.csv")

    ads.columns = ["date", "ads_index"]
    ads["date"] = ads["date"].str.replace(":", "-")
    ads["date"] = pd.to_datetime(ads["date"], format="%Y-%m-%d")
    if ads.empty:
        logging.error("ADS Index data was NOT loaded")
    logging.info("ADS Index data was loaded successfully")
    return ads


def get_sp500_data() -> pd.DataFrame:
    api_key = "74a4c86d8f52f8875f7e465e42f8e5de"
    fred = Fred(api_key=api_key)
    end_date = datetime.today().strftime("%Y-%m-%d")
    # Retrieve the S&P 500 data from FRED
    sp500_data = fred.get_series("SP500", end="end_date")
    # Convert to DataFrame for easier handling (optional)
    sp500 = pd.DataFrame(sp500_data, columns=["SP500"])
    sp500.reset_index(inplace=True)
    sp500.columns = ["date", "SP500"]

    if sp500.empty:
        logging.error("SP500 data was NOT loaded")
    logging.info("SP500 data was loaded successfully")

    return sp500


def get_fred_data():
    if storage_client is not None:  ## connect to GCS
        fred = get_fred_data_GCS()
    else:  ## local
        fred = get_fred_data_local()
    return fred


def get_fred_data_local() -> pd.DataFrame:
    ## read all files in the directory
    all_files = glob.glob("pipeline/airflow/dags/data/FRED_Variables/*.csv")
    fred = pd.read_csv(all_files[0])
    fred["DATE"] = pd.to_datetime(fred["DATE"], format="%Y-%m-%d")

    for file in all_files[1:]:
        df = pd.read_csv(file)
        df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d")
        fred = pd.merge(fred, df, on="DATE", how="outer")
    fred = fred.sort_values("DATE")
    fred["DATE"] = fred["DATE"].dt.strftime("%Y-%m-%d")

    all_cols = fred.columns
    new_cols = ["date"] + list(all_cols[1:])
    fred.columns = new_cols

    if fred.empty:
        logging.error("FRED data was NOT loaded")
    logging.info("FRED data was loaded successfully")

    return fred


def get_fred_data_GCS() -> pd.DataFrame:
    bucket_name = "stock_price_prediction_dataset"
    blob_name = "Data/pipeline/airflow/dags/data/FRED_Variables"
    # Note: Client.list_blobs requires at least package version 1.17.0.

    blobs = storage_client.list_blobs(bucket_name, prefix=blob_name)
    ## convert to list
    blobs = list(blobs)

    with blobs[0].open("r") as f:
        fred = f.read()  ## fred is a string
        fred = pd.read_csv(io.StringIO(fred), sep=",")
        fred["DATE"] = pd.to_datetime(fred["DATE"], format="%Y-%m-%d")

    for file in blobs[1:]:
        # option 1
        with file.open("r") as f:
            tmp_str = f.read()  ## fred is a string
            df = pd.read_csv(io.StringIO(tmp_str), sep=",")

        ## option 2
        # file_path = f"gs://stock_price_prediction_dataset/{file.name}"
        # with fs.open(file_path) as f:
        #     df = pd.read_csv(f)

        # convert DATE to date
        df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d")
        fred = pd.merge(fred, df, on="DATE", how="outer")
    fred = fred.sort_values("DATE")
    fred["DATE"] = fred["DATE"].dt.strftime("%Y-%m-%d")

    all_cols = fred.columns
    new_cols = ["date"] + list(all_cols[1:])
    fred.columns = new_cols

    if fred.empty:
        logging.error("FRED data was NOT loaded")
    logging.info("FRED data was loaded successfully")

    return fred


def merge_data(ticker_symbol: str) -> pd.DataFrame:
    # Fetch data
    yfinance = get_yfinance_data(ticker_symbol)
    fama_french = get_fama_french_data()
    ads_index = get_ads_index_data()
    sp500 = get_sp500_data()
    fred = get_fred_data()

    # Merge data
    df_list = [yfinance, fama_french, ads_index, sp500, fred]
    res = df_list[0]
    res["date"] = pd.to_datetime(res["date"], format="%Y-%m-%d")
    for df in df_list[1:]:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        res = pd.merge(res, df, on="date", how="outer")
    res = res.sort_values("date")
    res["date"] = res["date"].dt.strftime("%Y-%m-%d")
    res = res.loc[:, ~res.columns.str.contains("^Unnamed")]

    gcs_file_path = "Data/pipeline/airflow/dags/data/merged_original_dataset.csv"

    if storage_client is not None:
        bucket = storage_client.bucket("stock_price_prediction_dataset")
        blob = bucket.blob(gcs_file_path)
        blob.upload_from_string(res.to_csv(index=False), "text/csv")

    if res.empty:
        logging.error("Data was NOT merged")
    logging.info("Final Data was merged successfully")
    return res


if __name__ == "__main__":
    data = merge_data("GOOGL")
    data.to_csv("dags/data/merged_original_dataset.csv", index=False)
    print(data)
