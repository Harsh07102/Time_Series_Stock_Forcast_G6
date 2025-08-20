from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

def train_prophet(df, date_col='ds', target_col='y'):
    model = Prophet()
    model.fit(df[[date_col, target_col]])
    return model

def forecast_prophet(model, periods=30):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    fig = model.plot(forecast)
    plt.title("Prophet Forecast")

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], fig

def run_prophet_model(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Wrapper for training and forecasting with Prophet.
    Assumes df has a datetime index and a 'Close' column.
    """
    prophet_df = df.copy()
    prophet_df = prophet_df.reset_index()[['Date', 'Close']]
    prophet_df.columns = ['ds', 'y']

    model = train_prophet(prophet_df)
    forecast, _ = forecast_prophet(model, periods=horizon)

    forecast_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Forecast'})
    forecast_df = forecast_df.set_index('Date').tail(horizon)

    return forecast_df
