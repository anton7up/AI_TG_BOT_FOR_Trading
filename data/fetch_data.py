import yfinance as yf
import pandas as pd

def fetch_data(ticker='BTC-USD', period='5y', interval='1d'):
    df = yf.download(ticker, period=period, interval=interval)
    df = df[['Close']]
    df.dropna(inplace=True)
    return df
