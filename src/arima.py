import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def train_arima(series: pd.Series, order=(5, 1, 0)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def forecast_arima(model_fit, steps=30):
    forecast = model_fit.forecast(steps=steps)

    fig, ax = plt.subplots()
    ax.plot(forecast, label="ARIMA Forecast", color="orange")
    ax.set_title("ARIMA Forecast")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Predicted Value")
    ax.legend()

    return forecast, fig

def run_arima_model(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Wrapper for training and forecasting with ARIMA.
    Assumes df has a datetime index and a 'Close' column.
    """
    model_fit = train_arima(df["Close"])
    forecast, _ = forecast_arima(model_fit, steps=horizon)

    future_dates = pd.date_range(start=df.index[-1], periods=horizon + 1, freq="D")[1:]
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast.values
    }).set_index("Date")

    return forecast_df
