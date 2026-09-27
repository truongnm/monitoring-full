"""Register the image's trained WDBC model in the MLflow Model Registry."""

import json
import hashlib
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = os.getenv("MODEL_NAME", "wdbc-diagnosis")


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    frame = pd.read_csv(ROOT / "data/raw/wdbc.csv").dropna()
    frame = frame[frame["diagnosis"].isin({"M", "B"})].drop_duplicates("sample_id")
    card = json.loads((ROOT / "models/model_card.json").read_text())
    features = card["features"]
    x = frame[features]
    y = (frame["diagnosis"] == "M").astype(int)
    model_path = ROOT / "models/model.joblib"
    model = joblib.load(model_path)
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    minimum_auc = float(os.getenv("MIN_TEST_ROC_AUC", "0.90"))
    if metrics["roc_auc"] < minimum_auc:
        raise RuntimeError(
            f"Model ROC AUC {metrics['roc_auc']:.4f} is below promotion threshold {minimum_auc:.4f}"
        )

    client = MlflowClient(tracking_uri=tracking_uri)
    registered_models = client.search_registered_models(filter_string=f"name='{MODEL_NAME}'")
    if registered_models:
        production_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if production_versions and production_versions[0].tags.get("source_sha256") == model_sha256:
            print(
                f"{MODEL_NAME} Production already matches model artifact "
                f"(version {production_versions[0].version}); skipping registration"
            )
            return

    mlflow.set_experiment("wdbc-diagnosis")
    with mlflow.start_run(run_name=f"wdbc-{card['version']}") as run:
        mlflow.log_params({
            "algorithm": "StandardScaler + LogisticRegression",
            "max_iter": 5000,
            "random_state": 42,
            "train_rows": card["train_rows"],
            "sklearn_version": card["sklearn_version"],
            "split_random_state": 42,
            "test_size": 0.2,
        })
        mlflow.log_metrics({f"test_{name}": value for name, value in metrics.items()})
        mlflow.log_dict(card, "model_card.json")
        mlflow.log_dict({
            "labels": ["benign", "malignant"],
            "matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        }, "confusion_matrix.json")
        info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None,
            signature=infer_signature(x_train.head(5), model.predict(x_train.head(5))),
            input_example=x_train.head(3),
        )

    if not registered_models:
        client.create_registered_model(MODEL_NAME)
    current = client.create_model_version(
        name=MODEL_NAME,
        source=info.model_uri,
        run_id=run.info.run_id,
    )
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=current.version,
        key="source_sha256",
        value=model_sha256,
    )

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=current.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(
        f"Registered {MODEL_NAME} version {current.version} as Production "
        f"(run {run.info.run_id}, ROC AUC {metrics['roc_auc']:.5f}, artifact {info.model_uri})"
    )


if __name__ == "__main__":
    main()
