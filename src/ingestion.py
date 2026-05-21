import pandas as pd
import numpy as np
import os

RAW_PATH = os.path.join("data", "raw", "diabetes.csv")
INTERIM_PATH = os.path.join("data", "interim", "cleaned.csv")

def load_raw_data(path=RAW_PATH):
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape}")
    return df

def basic_cleaning(df):
    df = df.copy()
    columns_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in columns_with_zeros:
        df[col] = df[col].replace(0, np.nan)
    null_counts = df.isnull().sum()
    print(f"Null values after marking zeros:\n{null_counts[null_counts > 0]}")
    return df

def save_interim(df, path=INTERIM_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved interim data to {path}")

def run():
    df = load_raw_data()
    df_cleaned = basic_cleaning(df)
    save_interim(df_cleaned)

if __name__ == "__main__":
    run()
