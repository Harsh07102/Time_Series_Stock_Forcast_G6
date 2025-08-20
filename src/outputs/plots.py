import matplotlib.pyplot as plt
import pandas as pd

def plot_forecast(df_actual: pd.DataFrame, df_forecast: pd.DataFrame):
    """
    Plots actual vs forecasted values.
    Assumes both DataFrames have datetime index and 'Close' / 'Forecast' columns.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    if "Close" in df_actual.columns:
        ax.plot(df_actual.index, df_actual["Close"], label="Actual", color="blue")
    ax.plot(df_forecast.index, df_forecast["Forecast"], label="Forecast", color="orange")
    ax.set_title("Stock Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    return fig
