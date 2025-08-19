import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show_missing_values(df):
    """Display missing value counts and percentages."""
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing, 'Percent': missing_percent})
    return missing_df[missing_df['Missing Values'] > 0].sort_values(by='Percent', ascending=False)

def summary_statistics(df):
    """Return basic descriptive statistics."""
    return df.describe().T

def plot_correlation_matrix(df, figsize=(10, 8)):
    """Plot correlation heatmap for numeric features."""
    corr = df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

def plot_distributions(df, columns=None):
    """Plot histograms for selected columns."""
    if columns is None:
        columns = df.select_dtypes(include='number').columns
    df[columns].hist(figsize=(15, 10), bins=30, edgecolor='black')
    plt.tight_layout()
    plt.show()

def plot_time_series(df, date_col, value_col, title=None):
    """Plot time series data."""
    plt.figure(figsize=(12, 6))
    plt.plot(df[date_col], df[value_col], label=value_col)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title(title or f"{value_col} over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()
