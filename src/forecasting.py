import pandas as pd
from .arima import run_arima_model
from .Prophet import run_prophet_model

def run_forecast(df: pd.DataFrame, horizon: int, model_type: str = "ARIMA") -> pd.DataFrame:
    """
    Unified forecast interface for Streamlit app.
    
    Parameters:
    - df: Preprocessed DataFrame with datetime index and 'Close' column
    - horizon: Number of future days to forecast
    - model_type: 'ARIMA' or 'PROPHET'
    
    Returns:
    - forecast_df: DataFrame with future dates and predicted values
    """
    if model_type.upper() == "ARIMA":
        forecast_df = run_arima_model(df, horizon)
    elif model_type.upper() == "PROPHET":
        forecast_df = run_prophet_model(df, horizon)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Ensure forecast_df has datetime index and 'Forecast' column
    if not isinstance(forecast_df.index, pd.DatetimeIndex):
        forecast_df.index = pd.to_datetime(forecast_df.index)
    if "Forecast" not in forecast_df.columns:
        raise ValueError("Forecast output must contain a 'Forecast' column.")
    
    return forecast_df
