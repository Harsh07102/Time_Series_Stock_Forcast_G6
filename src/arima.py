import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def train_arima(series: pd.Series, order=(5, 1, 0)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit

def forecast_arima(model_fit, steps=30):
    # Generate forecast
    forecast = model_fit.forecast(steps=steps)

    # Create plot
    fig, ax = plt.subplots()
    ax.plot(forecast, label="ARIMA Forecast", color="orange")
    ax.set_title("ARIMA Forecast")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Predicted Value")
    ax.legend()

    # Return both forecast and figure
    return forecast, fig
