import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("diabetes_clinical_classification")

X_TEST_PATH = os.path.join("data", "processed", "X_test.csv")
Y_TEST_PATH = os.path.join("data", "processed", "y_test.csv")
REPORTS_DIR = os.path.join("reports")

def load_data():
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()
    print(f"Loaded test data: {X_test.shape}")
    return X_test, y_test

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        ll = log_loss(y_test, y_prob)
    except AttributeError:
        auc = 0.0
        ll = 0.0
    
    print(f"\n--- {model_name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Log Loss:  {ll:.4f}")
    print(report)
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plot_path = os.path.join(REPORTS_DIR, f"cm_{model_name.replace(' ', '_')}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    
    return acc, prec, rec, f1, auc, ll, plot_path

def run():
    X_test, y_test = load_data()
    experiment = mlflow.get_experiment_by_name("diabetes_clinical_classification")
    if experiment is None:
        print("No experiment found. Run train.py first.")
        return
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    all_metrics = {}
    for _, run_row in runs.iterrows():
        run_id = run_row["run_id"]
        model_type = run_row.get("tags.mlflow.runName", "unknown")
        model_uri = f"runs:/{run_id}/model"
        try:
            model = mlflow.sklearn.load_model(model_uri)
        except Exception:
            continue
        acc, prec, rec, f1, auc, ll, plot_path = evaluate_model(model, X_test, y_test, model_type)
        all_metrics[model_type] = {
            "test_accuracy": round(acc, 4),
            "test_precision": round(prec, 4),
            "test_recall": round(rec, 4),
            "test_f1_score": round(f1, 4),
            "test_auc_roc": round(auc, 4),
            "test_log_loss": round(ll, 4),
        }
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                "test_accuracy": acc,
                "test_precision": prec,
                "test_recall": rec,
                "test_f1_score": f1,
                "test_auc_roc": auc,
                "test_log_loss": ll
            })
            mlflow.log_artifact(plot_path)
        print(f"Metrics logged for {model_type}")
    import json
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

if __name__ == "__main__":
    run()
