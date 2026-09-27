"""
Train the model this tutorial serves. 

Run:  python scripts/train_model.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "wdbc.csv"
MODELS = ROOT / "models"

DROP = ["sample_id", "diagnosis"]


def main() -> None:
    frame = pd.read_csv(RAW).dropna()
    frame = frame[frame["diagnosis"].isin({"M", "B"})].drop_duplicates("sample_id")
    features = [c for c in frame.columns if c not in DROP]
    X = frame[features]
    y = (frame["diagnosis"] == "M").astype(int)      # 1 = malignant

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5000, random_state=42))
    model.fit(X_tr, y_tr)
    predictions = model.predict(X_te)
    probabilities = model.predict_proba(X_te)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_te, predictions)),
        "precision": float(precision_score(y_te, predictions, zero_division=0)),
        "recall": float(recall_score(y_te, predictions, zero_division=0)),
        "f1": float(f1_score(y_te, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, probabilities)),
    }
    matrix = confusion_matrix(y_te, predictions, labels=[0, 1]).tolist()

    MODELS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / "model.joblib")
    (MODELS / "model_card.json").write_text(json.dumps({
        "version": "1.0.0",
        "model_name": "wdbc-diagnosis",
        "algorithm": "StandardScaler + LogisticRegression",
        "sklearn_version": sklearn.__version__,
        "features": features,
        "feature_count": len(features),
        "train_rows": len(X_tr),
        "test_rows": len(X_te),
        "test_metrics": {name: round(value, 6) for name, value in metrics.items()},
        "test_roc_auc": round(metrics["roc_auc"], 6),
        "confusion_matrix_labels": ["benign", "malignant"],
        "confusion_matrix": matrix,
        "split": {"test_size": 0.2, "stratified": True, "random_state": 42},
        "positive_class": "M (malignant)",
    }, indent=2))
    print(f"trained on {len(X_tr)} rows, evaluated on {len(X_te)} rows")
    print("test metrics: " + json.dumps(metrics, sort_keys=True))
    print(f"confusion matrix [benign, malignant]: {matrix}")
    print(f"saved -> models/model.joblib  and  models/model_card.json")


if __name__ == "__main__":
    main()
