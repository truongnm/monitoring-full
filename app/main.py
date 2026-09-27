"""
Breast cancer diagnosis API — the service Tutorial 04 watches.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from app.metrics import (ERRORS, LATENCY, MALIGNANT_SHARE, MODEL_INFO,
                         MODEL_LOADED, PREDICTIONS, TRAP)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
THRESHOLD = 0.5
WINDOW = 200

app = FastAPI(title="WDBC diagnosis API", version="1.0.0")

state: Dict[str, object] = {"model": None, "card": None, "model_version": None}
recent: deque = deque(maxlen=WINDOW)
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    sample_id: str = Field(..., examples=["WDBC-0001"])
    features: Dict[str, float]


@app.on_event("startup")
def load_model() -> bool:
    path = MODELS / "model.joblib"
    backend = os.getenv("MODEL_BACKEND", "joblib").lower()
    card_path = MODELS / "model_card.json"
    if not card_path.exists() or (backend != "mlflow" and not path.exists()):
        MODEL_LOADED.set(0)
        return False

    card = json.loads(card_path.read_text())
    if backend == "mlflow":
        import mlflow
        import mlflow.sklearn
        from mlflow.tracking import MlflowClient

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        model_name = os.getenv("MODEL_NAME", "wdbc-diagnosis")
        stage = os.getenv("MODEL_STAGE", "Production")
        mlflow.set_tracking_uri(tracking_uri)
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
        versions = MlflowClient(tracking_uri=tracking_uri).get_latest_versions(
            model_name, stages=[stage]
        )
        model_version = versions[0].version if versions else stage
    else:
        model = joblib.load(path)
        model_version = card["version"]

    # Load and validate the new model before swapping it in, so a failed reload
    # leaves the currently serving model available.
    state["card"] = card
    state["model_version"] = model_version
    state["model"] = model
    MODEL_LOADED.set(1)
    MODEL_INFO.labels(version=str(model_version),
                      sklearn_version=card["sklearn_version"]).set(1)
    return True


@app.get("/health")
def health() -> dict:
    # The health check reports what it can actually do, not merely that the
    # process is alive. A container that answers 200 while serving 503s to
    # every real request is worse than one that is honestly down.
    return {"status": "ok" if state["model"] else "degraded",
            "model_loaded": state["model"] is not None,
            "version": state["model_version"]}


@app.get("/model/info")
def model_info() -> dict:
    card = state["card"] or {}
    return {
        "model_loaded": state["model"] is not None,
        "model_name": os.getenv("MODEL_NAME", "wdbc-diagnosis"),
        "model_version": state["model_version"],
        "backend": os.getenv("MODEL_BACKEND", "joblib").lower(),
        "sklearn_version": card.get("sklearn_version"),
        "feature_count": len(card.get("features", [])),
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI") if os.getenv("MODEL_BACKEND", "joblib").lower() == "mlflow" else None,
    }


@app.post("/model/reload")
def reload_model() -> dict:
    try:
        if not load_model():
            raise RuntimeError("model artifact or model card is missing")
    except Exception as exc:
        logger.exception("Could not reload model")
        raise HTTPException(status_code=503, detail="could not load the configured model") from exc
    return {"status": "reloaded", "model_version": state["model_version"]}


def capture_evidently(features: Dict[str, float], probability: float,
                      model_version: object) -> None:
    url = os.getenv("EVIDENTLY_CAPTURE_URL", "").strip()
    if not url:
        return
    try:
        response = httpx.post(
            url,
            json={"features": features, "prediction": probability,
                  "model_version": str(model_version)},
            timeout=2.0,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        logger.exception("Could not send prediction to the Evidently service")


@app.post("/predict")
def predict(req: PredictRequest, background_tasks: BackgroundTasks) -> dict:
    if state["model"] is None:
        ERRORS.labels(reason="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="model not loaded")

    features: List[str] = state["card"]["features"]
    missing = [f for f in features if f not in req.features]
    if missing:
        ERRORS.labels(reason="missing_features").inc()
        raise HTTPException(status_code=422,
                            detail=f"missing features: {missing[:3]}")

    # The timer wraps only the work being measured. Wrap the whole request and
    # you are also measuring FastAPI's parsing, which is not what the number
    # is supposed to mean.
    start = time.perf_counter()
    frame = pd.DataFrame([[req.features[f] for f in features]], columns=features)
    probability = float(state["model"].predict_proba(frame)[0, 1])
    LATENCY.observe(time.perf_counter() - start)

    outcome = "malignant" if probability >= THRESHOLD else "benign"
    labels = {"outcome": outcome}
    if TRAP:
        # Section 6. One new time series per patient, forever.
        labels["sample_id"] = req.sample_id
    PREDICTIONS.labels(**labels).inc()

    recent.append(1 if outcome == "malignant" else 0)
    MALIGNANT_SHARE.set(sum(recent) / len(recent))
    background_tasks.add_task(
        capture_evidently, req.features, probability, state["model_version"]
    )

    return {"sample_id": req.sample_id, "probability": round(probability, 6),
            "outcome": outcome, "threshold": THRESHOLD,
            "model_version": state["model_version"]}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    # Plain text in Prometheus exposition format. Open it in a browser once and
    # read it -- it is the entire contract between your service and Prometheus,
    # and it is simpler than people expect.
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
