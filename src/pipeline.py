import pandas as pd
from Feature_Engineering import (
    add_lag_features,
    add_rolling_features,
    add_date_features,
    add_volatility,
    add_momentum
)
from Prophet import train_prophet, forecast_prophet
from arima import train_arima, forecast_arima
from utils import setup_logging, set_seed

def run_pipeline():
    setup_logging()
    set_seed()

    # Load preprocessed data
    try:
        df = pd.read_csv('processed_stock_data.csv')

        print("✅ Loaded processed_stock_data.csv")
    except FileNotFoundError:
        print("❌ File not found: Run preprocessing.py first to generate processed_stock_data.csv")
        return

    # Feature Engineering
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date')
        df = df.set_index('Date')

        df = add_lag_features(df, column='Close')
        df = add_rolling_features(df, column='Close')
        df = add_date_features(df, date_col=df.index.to_series())
        df = add_volatility(df, column='Close')
        df = add_momentum(df, column='Close')

        df = df.dropna()
        print("🔧 Feature engineering complete")
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return

    # Prophet Forecasting
    try:
        prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        prophet_model = train_prophet(prophet_df)
        prophet_forecast = forecast_prophet(prophet_model, periods=30)
        print("📈 Prophet forecast:")
        print(prophet_forecast.tail())
    except Exception as e:
        print(f"⚠️ Prophet forecasting failed: {e}")

    # ARIMA Forecasting
    try:
        arima_model = train_arima(df['Close'], order=(5, 1, 0))
        arima_forecast = forecast_arima(arima_model, steps=30)
        print("📉 ARIMA forecast:")
        print(arima_forecast.tail())
    except Exception as e:
        print(f"⚠️ ARIMA forecasting failed: {e}")

if __name__ == "__main__":
    run_pipeline()
