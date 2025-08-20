import pandas as pd
from Feature_Engineering import (
    add_lag_features,
    add_rolling_features,
    add_date_features,
    add_volatility,
    add_momentum
)
from utils import setup_logging, set_seed

# Optional: Only include these if the files exist in src/
try:
    from Model_comparison import compare_models
    from Prophet import train_prophet, forecast_prophet
    from arima import train_arima, forecast_arima
    prophet_enabled = True
    arima_enabled = True
except ImportError:
    print("⚠️ Optional modules not found: Prophet, ARIMA, or Model_comparison")
    prophet_enabled = False
    arima_enabled = False

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def run_pipeline():
    setup_logging()
    set_seed()

    # Load preprocessed data
    try:
        df = pd.read_csv('../Data/procesed/processed_stock_data.csv')
        print(" Loaded processed_stock_data.csv")
    except FileNotFoundError:
        print("File not found: Make sure preprocessing.py has generated processed_stock_data.csv")
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
        X = df.drop(columns=['Close'])
        y = df['Close']
        print("🔧 Features created successfully")
    except Exception as e:
        print(f"Feature engineering failed: {e}")
        return

    # Model Comparison
    if prophet_enabled and arima_enabled:
        try:
            models = {
                'LinearRegression': LinearRegression(),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            results = compare_models(models, X, y)
            print(" Model Comparison Results:")
            for model, metrics in results.items():
                print(f"{model}: RMSE={metrics['RMSE']}, STD={metrics['STD']}")
        except Exception as e:
            print(f"Model comparison failed: {e}")

        # Prophet Forecasting
        try:
            prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
            prophet_model = train_prophet(prophet_df)
            prophet_forecast = forecast_prophet(prophet_model, periods=30)
            print("Prophet forecast generated")
            print(prophet_forecast.tail())
        except Exception as e:
            print(f"⚠️ Prophet forecasting failed: {e}")

        # ARIMA Forecasting
        try:
            arima_model = train_arima(df['Close'], order=(5,1,0))
            arima_forecast = forecast_arima(arima_model, steps=30)
            print("ARIMA forecast generated")
            print(arima_forecast.tail())
        except Exception as e:
            print(f"⚠️ ARIMA forecasting failed: {e}")
    else:
        print("⚠️ Skipping model comparison and forecasting due to missing modules.")

if __name__ == "__main__":
    run_pipeline()
