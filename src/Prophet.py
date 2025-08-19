from prophet import Prophet
import pandas as pd

def train_prophet(df, date_col='ds', target_col='y'):
    model = Prophet()
    model.fit(df[[date_col, target_col]])
    return model

def forecast_prophet(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
