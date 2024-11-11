import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
import glob
from datetime import datetime
import os
import logging

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


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
    ff = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/fama_french.csv"))
    ff["Date"] = pd.to_datetime(ff["Date"], format="%Y%m%d")
    all_cols = ff.columns
    new_cols = ["date"] + list(all_cols[1:])
    ff.columns = new_cols
    if ff.empty:
        logging.error("Fama French data was NOT loaded")
    logging.info("Fama French data was loaded successfully")

    return ff


def get_ads_index_data() -> pd.DataFrame:
    ads = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/ADS_index.csv"))
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


def get_fred_data() -> pd.DataFrame:
    path = os.path.join(os.path.dirname(__file__), "../data/FRED_Variables") + "/*.csv"
    # print("1900 path", path)
    files = glob.glob(path)
    # print("files", files)
    fred = pd.read_csv(files[0])
    fred["DATE"] = pd.to_datetime(fred["DATE"], format="%Y-%m-%d")
    for file in files[1:]:
        # convert DATE to date
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

    if res.empty:
        logging.error("Data was NOT merged")
    logging.info("Final Data was merged successfully")
    return res


if __name__ == "__main__":
    data = merge_data("GOOGL")
    data.to_csv("dags/data/merged_original_dataset.csv", index=False)
    print(data)
