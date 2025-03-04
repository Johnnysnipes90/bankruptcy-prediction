import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import os
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def load_data():
    """Load CSV file and convert to pandas DataFrame."""
    df = pd.read_csv(RAW_DATA_PATH)
    return df

def preprocess_data(df):
    """Handle missing values and return cleaned DataFrame."""
    df.fillna(df.median(), inplace=True)
    return df

def split_data(df, target='bankrupt'):
    """Splits data into training and testing sets."""
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def resample_data(X_train, y_train, strategy='over'):
    """Resample training data (oversampling or undersampling)."""
    if strategy == 'under':
        sampler = RandomUnderSampler(random_state=42)
    else:
        sampler = RandomOverSampler(random_state=42)
    return sampler.fit_resample(X_train, y_train)

def save_data(df):
    """Save processed data to CSV."""
    df.to_csv(PROCESSED_DATA_PATH, index=False)

if __name__ == "__main__":
    df = load_data()
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    save_data(pd.concat([X_train, y_train], axis=1))
    print("Data preprocessing complete and saved!")
