import os
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# Timestamp for versioned outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M")

# Create output folders
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# Setup logging
logging.basicConfig(filename=f"outputs/logs/pipeline_run_{timestamp}.log",
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def save_plot(fig, filename):
    fig.savefig(f"outputs/plots/{filename}")
    logging.info(f"Saved plot: {filename}")

def save_metrics(metrics_dict, filename):
    with open(f"outputs/metrics/{filename}", "w") as f:
        json.dump(metrics_dict, f, indent=4)
    logging.info(f"Saved metrics: {filename}")

def save_report(metrics_dict, filename="summary.md"):
    with open(f"outputs/reports/{filename}", "w") as f:
        f.write("# 📊 Forecast Summary Report\n\n")
        for model_name, metrics in metrics_dict.items():
            f.write(f"## {model_name}\n")
            for metric, value in metrics.items():
                f.write(f"- **{metric}**: {value:.4f}\n")
            f.write("\n")
    logging.info(f"Saved report: {filename}")

def run_pipeline():
    setup_logging()
    set_seed()

    # Load preprocessed data
    try:
        df = pd.read_csv('processed_stock_data.csv')
        print("✅ Loaded processed_stock_data.csv")
        logging.info("Loaded processed_stock_data.csv")
    except FileNotFoundError:
        print("❌ File not found: Run preprocessing.py first to generate processed_stock_data.csv")
        logging.error("processed_stock_data.csv not found")
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
        logging.info("Feature engineering complete")
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        logging.error(f"Feature engineering failed: {e}")
        return

    all_metrics = {}

    # Prophet Forecasting
    try:
        prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        prophet_model = train_prophet(prophet_df)
        prophet_forecast, prophet_fig = forecast_prophet(prophet_model, periods=30)

        save_plot(prophet_fig, f"prophet_forecast_{timestamp}.png")

        prophet_metrics = {
            "MAE": mean_absolute_error(prophet_df["y"], prophet_forecast["yhat"]),
            "RMSE": mean_squared_error(prophet_df["y"], prophet_forecast["yhat"], squared=False)
        }
        save_metrics(prophet_metrics, f"prophet_metrics_{timestamp}.json")
        all_metrics["Prophet"] = prophet_metrics

        # Residual plot
        residuals = prophet_df["y"] - prophet_forecast["yhat"]
        fig, ax = plt.subplots()
        ax.plot(residuals)
        ax.set_title("Prophet Residuals")
        save_plot(fig, f"prophet_residuals_{timestamp}.png")

        print("📈 Prophet forecast:")
        print(prophet_forecast.tail())
        logging.info("Prophet forecasting complete")
    except Exception as e:
        print(f"⚠️ Prophet forecasting failed: {e}")
        logging.error(f"Prophet forecasting failed: {e}")

    # ARIMA Forecasting
    try:
        arima_model = train_arima(df['Close'], order=(5, 1, 0))
        arima_forecast, arima_fig = forecast_arima(arima_model, steps=30)

        save_plot(arima_fig, f"arima_forecast_{timestamp}.png")

        arima_metrics = {
            "MAE": mean_absolute_error(df["Close"][-30:], arima_forecast),
            "RMSE": mean_squared_error(df["Close"][-30:], arima_forecast, squared=False)
        }
        save_metrics(arima_metrics, f"arima_metrics_{timestamp}.json")
        all_metrics["ARIMA"] = arima_metrics

        # Residual plot
        residuals = df["Close"][-30:] - arima_forecast
        fig, ax = plt.subplots()
        ax.plot(residuals)
        ax.set_title("ARIMA Residuals")
        save_plot(fig, f"arima_residuals_{timestamp}.png")

        print("📉 ARIMA forecast:")
        print(arima_forecast.tail())
        logging.info("ARIMA forecasting complete")
    except Exception as e:
        print(f"⚠️ ARIMA forecasting failed: {e}")
        logging.error(f"ARIMA forecasting failed: {e}")

    # Save summary report
    try:
        save_report(all_metrics, filename=f"summary_{timestamp}.md")
    except Exception as e:
        logging.error(f"Failed to save summary report: {e}")

if __name__ == "__main__":
    run_pipeline()
