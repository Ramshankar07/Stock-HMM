"""Functions for data downloading and preprocessing"""

import numpy as np
import pandas as pd
import requests
import streamlit as st

def get_stock_data(api_key: str, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical stock price data from Alpha Vantage API.
    """
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'apikey': api_key,
        'outputsize': 'full'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'Time Series (Daily)' not in data:
        print(f"Error: {data}")
        return None
        
    time_series = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df[(df.index >= pd.to_datetime(start_date)) 
            & (df.index <= pd.to_datetime(end_date))]
    return df

@st.cache_data
def make_stock_price_df(start_date: str, end_date: str) -> pd.DataFrame:
    """Return dataframe with stock price timeseries with date as index."""
    api_key = st.secrets["api"]["key"]
    
    df_aapl = get_stock_data(api_key, 'AAPL', start_date, end_date)
    df_aapl = df_aapl[['close']].rename(columns={'close': 'AAPL'})
    
    df_tsla = get_stock_data(api_key, 'TSLA', start_date, end_date)
    df_tsla = df_tsla[['close']].rename(columns={'close': 'TSLA'})
    
    df_nvda = get_stock_data(api_key, 'NVDA', start_date, end_date)
    df_nvda = df_nvda[['close']].rename(columns={'close': 'NVDA'})
    
    df = pd.concat([df_aapl, df_tsla, df_nvda], axis=1)
    df = df.ffill()  # Impute missing values (forward filling)
    df = df.dropna()
    df = df.sort_index()
    return df

@st.cache_data
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert prices to (log-)returns."""
    return pd.DataFrame(
        data=np.log(df.iloc[:-1].values / df.iloc[1:].values),
        index=df.iloc[1:].index,
        columns=df.columns)