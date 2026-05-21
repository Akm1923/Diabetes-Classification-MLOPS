import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("diabetes_clinical_classification")

X_TRAIN_PATH = os.path.join("data", "processed", "X_train.csv")
Y_TRAIN_PATH = os.path.join("data", "processed", "y_train.csv")
MODELS_DIR = os.path.join("models")

FEATURE_NAMES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

def load_data():
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
    print(f"Loaded training data: {X_train.shape}")
    return X_train, y_train

def log_common_params(X_train, y_train):
    mlflow.log_param("n_samples", len(X_train))
    mlflow.log_param("n_features", X_train.shape[1])

def train_logistic_regression(X_train, y_train):
    with mlflow.start_run(run_name="LogisticRegression"):
        params = {"C": 10, "solver": "lbfgs", "max_iter": 1000, "penalty": "l2"}
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        log_common_params(X_train, y_train)
        mlflow.log_params(params)
        mlflow.log_param("model_type", "LogisticRegression")
        train_score = model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.sklearn.log_model(model, "model")
        print(f"LogisticRegression train accuracy: {train_score:.4f}")
    return model

def train_svc(X_train, y_train):
    with mlflow.start_run(run_name="SVC"):
        params = {"C": 100, "kernel": "rbf", "gamma": "scale"}
        model = SVC(**params)
        model.fit(X_train, y_train)
        log_common_params(X_train, y_train)
        mlflow.log_params(params)
        mlflow.log_param("model_type", "SVC")
        train_score = model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.sklearn.log_model(model, "model")
        print(f"SVC train accuracy: {train_score:.4f}")
    return model

def train_random_forest(X_train, y_train):
    with mlflow.start_run(run_name="RandomForest"):
        params = {"n_estimators": 100, "random_state": 42, "max_depth": None,
                  "min_samples_split": 2, "min_samples_leaf": 1, "criterion": "gini"}
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        log_common_params(X_train, y_train)
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForest")
        train_score = model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_score)
        importances = model.feature_importances_
        for name, imp in zip(FEATURE_NAMES, importances):
            mlflow.log_metric(f"feature_importance_{name}", float(imp))
        mlflow.sklearn.log_model(model, "model")
        print(f"RandomForest train accuracy: {train_score:.4f}")
    return model

def train_decision_tree(X_train, y_train):
    with mlflow.start_run(run_name="DecisionTree"):
        params = {"random_state": 42, "max_depth": None, "min_samples_split": 2,
                  "min_samples_leaf": 1, "criterion": "gini"}
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        log_common_params(X_train, y_train)
        mlflow.log_params(params)
        mlflow.log_param("model_type", "DecisionTree")
        train_score = model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_score)
        importances = model.feature_importances_
        for name, imp in zip(FEATURE_NAMES, importances):
            mlflow.log_metric(f"feature_importance_{name}", float(imp))
        mlflow.sklearn.log_model(model, "model")
        print(f"DecisionTree train accuracy: {train_score:.4f}")
    return model

def train_gradient_boosting(X_train, y_train):
    with mlflow.start_run(run_name="GradientBoosting"):
        params = {"random_state": 42, "n_estimators": 100, "learning_rate": 0.1,
                  "max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1,
                  "subsample": 1.0, "loss": "log_loss"}
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        log_common_params(X_train, y_train)
        mlflow.log_params(params)
        mlflow.log_param("model_type", "GradientBoosting")
        train_score = model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_score)
        importances = model.feature_importances_
        for name, imp in zip(FEATURE_NAMES, importances):
            mlflow.log_metric(f"feature_importance_{name}", float(imp))
        mlflow.sklearn.log_model(model, "model")
        print(f"GradientBoosting train accuracy: {train_score:.4f}")
    return model

def run():
    X_train, y_train = load_data()
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = {}
    models["LogisticRegression"] = train_logistic_regression(X_train, y_train)
    models["SVC"] = train_svc(X_train, y_train)
    models["RandomForest"] = train_random_forest(X_train, y_train)
    models["DecisionTree"] = train_decision_tree(X_train, y_train)
    models["GradientBoosting"] = train_gradient_boosting(X_train, y_train)
    print("All models trained successfully.")

if __name__ == "__main__":
    run()
