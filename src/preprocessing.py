import pandas as pd
import os

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data from a given filepath."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def convert_and_sort_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'Date' column to datetime and sort the DataFrame."""
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    df = df.set_index('Date')
    return df

def validate_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure key columns are numeric and drop rows with missing values."""
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=numeric_cols)
    return df

def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from the datetime index."""
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    df['Weekday'] = df.index.weekday
    return df

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using forward and backward fill."""
    return df.ffill().bfill()

def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """Save cleaned DataFrame to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

def preprocess_pipeline(input_path: str, output_path: str) -> None:
    """
    Full preprocessing pipeline from raw CSV to cleaned output file.
    Preserves original behavior for CLI or script-based execution.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_input_path = os.path.join(script_dir, input_path)

    project_root = os.path.dirname(script_dir)
    target_data_dir = os.path.join(project_root, 'Data', 'processed')
    full_output_path = os.path.join(target_data_dir, output_path)

    df = load_raw_data(full_input_path)
    df = convert_and_sort_dates(df)
    df = validate_numeric_columns(df)
    df = fill_missing_values(df)
    df = engineer_time_features(df)
    save_cleaned_data(df, full_output_path)
    print(f"Preprocessing complete. Saved to {full_output_path}")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Modular preprocessing function for use in Streamlit or other apps.
    Accepts a DataFrame and returns a cleaned version.
    """
    df = convert_and_sort_dates(df)
    df = validate_numeric_columns(df)
    df = fill_missing_values(df)
    df = engineer_time_features(df)
    return df

if __name__ == "__main__":
    input_path = "ITC_stock_data.csv"
    output_path = "processed_stock_data.csv"
    preprocess_pipeline(input_path, output_path)
