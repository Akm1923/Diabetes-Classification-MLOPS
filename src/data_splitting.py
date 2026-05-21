import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

PROCESSED_PATH = os.path.join("data", "processed", "engineered.csv")
X_TRAIN_PATH = os.path.join("data", "processed", "X_train.csv")
X_TEST_PATH = os.path.join("data", "processed", "X_test.csv")
Y_TRAIN_PATH = os.path.join("data", "processed", "y_train.csv")
Y_TEST_PATH = os.path.join("data", "processed", "y_test.csv")

def load_engineered(path=PROCESSED_PATH):
    df = pd.read_csv(path)
    print(f"Loaded engineered data: {df.shape}")
    return df

def split_and_balance(df, target_col="Outcome", test_size=0.10, random_state=50):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"Before SMOTE: {y.value_counts().to_dict()}")
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=test_size, random_state=random_state
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def save_splits(X_train, X_test, y_train, y_test):
    os.makedirs(os.path.dirname(X_TRAIN_PATH), exist_ok=True)
    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)
    print("Saved train/test splits to data/processed/")

def run():
    df = load_engineered()
    X_train, X_test, y_train, y_test = split_and_balance(df)
    save_splits(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    run()
