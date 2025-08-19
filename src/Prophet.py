from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

def train_prophet(df, date_col='ds', target_col='y'):
    model = Prophet()
    model.fit(df[[date_col, target_col]])
    return model

def forecast_prophet(model, periods=30):
    # Generate future dataframe
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Create forecast plot
    fig = model.plot(forecast)
    plt.title("Prophet Forecast")

    # Return both forecast and figure
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], fig
