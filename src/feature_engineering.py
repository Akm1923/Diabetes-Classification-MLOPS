import pandas as pd
import numpy as np
import os

INTERIM_PATH = os.path.join("data", "interim", "cleaned.csv")
PROCESSED_PATH = os.path.join("data", "processed", "engineered.csv")

def load_interim(path=INTERIM_PATH):
    df = pd.read_csv(path)
    print(f"Loaded interim data: {df.shape}")
    return df

def remove_outliers_iqr(df, features):
    df = df.copy()
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_mask = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        df.loc[outlier_mask, feature] = np.nan
        print(f"{feature}: flagged {outlier_mask.sum()} outliers")
    return df

def impute_median(df):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"{col}: imputed with median {median_val:.2f}")
    return df

def standard_scale(df, exclude_cols=["Outcome"]):
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
    return df

def save_processed(df, path=PROCESSED_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved engineered data to {path}")

def run():
    df = load_interim()
    df = remove_outliers_iqr(df, features=["Insulin", "DiabetesPedigreeFunction"])
    df = impute_median(df)
    df = standard_scale(df)
    save_processed(df)

if __name__ == "__main__":
    run()
