import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def train_arima(series: pd.Series, order=(5, 1, 0)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def forecast_arima(model_fit, steps=30):
    forecast = model_fit.forecast(steps=steps)
    return forecast
