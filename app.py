import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing import clean_data
from src.forecasting import run_forecast
from src.outputs.plots import plot_forecast


st.set_page_config(page_title="Stock Forecasting App", layout="wide")

st.title(" Modular Stock Forecasting Dashboard")

# Sidebar controls
st.sidebar.header("Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload stock CSV", type=["csv"])
forecast_days = st.sidebar.slider("Forecast horizon (days)", 1, 30, 7)

# Main logic
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Preprocessing
    try:
        df_clean = clean_data(df)
        st.success(" Data cleaned successfully.")
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        st.stop()

    # Forecasting
    try:
        forecast_df = run_forecast(df_clean, forecast_days)
        st.subheader("Forecast Output")
        st.dataframe(forecast_df.tail())
    except Exception as e:
        st.error(f" Forecasting failed: {e}")
        st.stop()

    # Plotting
    try:
        fig = plot_forecast(df_clean, forecast_df)
        st.subheader("Forecast Visualization")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"⚠️ Plotting issue: {e}")
else:
    st.info("Please upload a CSV file to begin.")
