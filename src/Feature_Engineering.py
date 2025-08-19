import pandas as pd
import numpy as np

def add_lag_features(df, column, lags=[1, 2, 3]):
    """Add lag features for a given column."""
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df

def add_rolling_features(df, column, windows=[3, 7, 14]):
    """Add rolling mean and std features."""
    for window in windows:
        df[f"{column}_roll_mean_{window}"] = df[column].rolling(window).mean()
        df[f"{column}_roll_std_{window}"] = df[column].rolling(window).std()
    return df

def add_date_features(df, date_col):
    """Extract date-based features."""
    df['day'] = df[date_col].dt.day
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    df['dayofweek'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df[date_col].dt.dayofweek >= 5
    return df

def add_volatility(df, column, window=5):
    """Add rolling volatility (standard deviation)."""
    df[f"{column}_volatility_{window}"] = df[column].rolling(window).std()
    return df

def add_momentum(df, column, window=5):
    """Add momentum indicator (difference from rolling mean)."""
    roll_mean = df[column].rolling(window).mean()
    df[f"{column}_momentum_{window}"] = df[column] - roll_mean
    return df
