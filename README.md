# Diabetes Classification - MLOps Pipeline

Production-grade ML pipeline for Pima Indians Diabetes classification, refactored from a monolithic notebook into modular MLOps components.

## Project Structure

```
├── data/
│   ├── raw/                    # Original diabetes.csv (tracked by DVC)
│   ├── interim/                # Cleaned data (zeros replaced with NaN)
│   └── processed/              # Engineered features + train/test splits (tracked by DVC)
├── notebooks/
│   └── eda.ipynb               # Detailed EDA with original plots and observations
├── src/
│   ├── ingestion.py            # Load raw data, basic cleaning
│   ├── feature_engineering.py  # Outlier removal (IQR), median imputation, z-score scaling
│   ├── data_splitting.py       # SMOTE balancing, train/test split
│   ├── train.py                # Sklearn models with MLflow tracking
│   └── evaluate.py             # Classification metrics, confusion matrix plots
├── requirements.txt
└── README.md
```

## Pipeline Execution Order

```bash
# 1. Ingestion
python src/ingestion.py

# 2. Feature Engineering
python src/feature_engineering.py

# 3. Data Splitting
python src/data_splitting.py

# 4. Training (with MLflow)
python src/train.py

# 5. Evaluation (with MLflow)
python src/evaluate.py
```

## DVC Setup

```bash
# Initialize DVC
dvc init

# Track data directories
dvc add data/raw
dvc add data/interim
dvc add data/processed

# Commit .dvc files to Git
git add data/raw.dvc data/interim.dvc data/processed.dvc .gitignore
git commit -m "Track data with DVC"
```

## MLflow

Experiments are logged under `diabetes_clinical_classification`. View with:

```bash
mlflow ui
```

## Models Trained

- Logistic Regression (C=10)
- SVC (kernel=rbf, C=100)
- Random Forest Classifier
- Decision Tree Classifier
- Gradient Boosting Classifier

## Tech Stack

- Pandas, NumPy — data manipulation
- Scikit-Learn — modeling & evaluation
- DVC — data version control
- MLflow — experiment tracking & model registry
- imbalanced-learn (SMOTE) — class balancing
