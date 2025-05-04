import yfinance as yf
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import numpy as np
import plotly.graph_objects as go


def fetch_stock_data(ticker, start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)

    data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        raise ValueError("No stock data available for the given period.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns]

    return data


def train_prophet_model(data, periods=365):
    close_column = next((col for col in data.columns if 'Close' in col), 'Close')

    df = data.reset_index()[['Date', close_column]]
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    df = df.sort_values('ds').dropna()

    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Calculate RMSE
    merged = df.merge(forecast, on='ds', how='inner')
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))

    return model, forecast, rmse
