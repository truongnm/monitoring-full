"""Collect WDBC prediction features and report feature drift with Evidently."""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REFERENCE_PATH = Path(os.getenv("REFERENCE_DATA_PATH", "/app/reference/wdbc.csv"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "/app/reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MAX_SAMPLES = 10000

DRIFT_DETECTED = Gauge("evidently_data_drift_detected", "1 when feature drift is detected")
DRIFT_SCORE = Gauge("evidently_drift_score", "Share of WDBC features detected as drifted")
DRIFTED_FEATURES = Gauge("evidently_drifted_features_count", "Number of drifted features")
FEATURE_DRIFT = Gauge("evidently_feature_drift", "Drift status by feature", ["feature_name"])
CAPTURE_TOTAL = Counter("evidently_captured_predictions_total", "Predictions received for drift analysis")
ANALYSIS_TOTAL = Counter("evidently_analysis_total", "Completed Evidently analyses")
ANALYSIS_DURATION = Histogram("evidently_analysis_duration_seconds", "Duration of Evidently analysis")
LAST_ANALYSIS = Gauge("evidently_last_analysis_timestamp_seconds", "Unix time of the last analysis")


class CaptureRequest(BaseModel):
    features: dict[str, float]
    prediction: float | None = None
    model_version: str | None = None


class AnalyzeRequest(BaseModel):
    window_size: int = Field(default=200, ge=10, le=MAX_SAMPLES)
    drift_share_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


app = FastAPI(title="WDBC Evidently Drift Monitor", version="1.0.0")
lock = threading.Lock()
production_samples: list[dict[str, Any]] = []
reference = pd.DataFrame()
feature_columns: list[str] = []
last_analysis: str | None = None


@app.on_event("startup")
def load_reference_data() -> None:
    global reference, feature_columns
    if not REFERENCE_PATH.exists():
        raise RuntimeError(f"Reference WDBC data not found: {REFERENCE_PATH}")
    frame = pd.read_csv(REFERENCE_PATH).dropna()
    feature_columns = [
        column for column in frame.columns
        if column not in {"sample_id", "diagnosis"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not feature_columns:
        raise RuntimeError("The WDBC reference dataset has no numeric features")
    reference = frame[feature_columns].copy()
    logger.info("Loaded %d reference rows and %d features", len(reference), len(feature_columns))


@app.get("/")
def root() -> dict[str, Any]:
    return {"service": "WDBC Evidently Drift Monitor", "endpoints": ["/health", "/capture", "/analyze", "/reports", "/metrics"]}


@app.get("/health")
def health() -> dict[str, Any]:
    with lock:
        count = len(production_samples)
    return {
        "status": "healthy" if not reference.empty else "degraded",
        "reference_samples": len(reference),
        "production_samples": count,
        "features": len(feature_columns),
        "last_analysis": last_analysis,
        "reports": len(list(REPORTS_DIR.glob("*.html"))),
    }


@app.post("/capture")
def capture(data: CaptureRequest) -> dict[str, Any]:
    missing = [column for column in feature_columns if column not in data.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing WDBC features: {missing[:3]}")
    row: dict[str, Any] = {column: data.features[column] for column in feature_columns}
    row.update({"prediction": data.prediction, "model_version": data.model_version,
                "timestamp": datetime.now(timezone.utc).isoformat()})
    with lock:
        production_samples.append(row)
        if len(production_samples) > MAX_SAMPLES:
            del production_samples[:len(production_samples) - MAX_SAMPLES]
        count = len(production_samples)
    CAPTURE_TOTAL.inc()
    return {"status": "captured", "production_samples": count}


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    global last_analysis
    with lock:
        rows = production_samples[-request.window_size:]
    if len(rows) < request.window_size:
        return {
            "status": "waiting_for_samples",
            "required": request.window_size,
            "available": len(rows),
        }

    current = pd.DataFrame(rows)[feature_columns]
    started = time.perf_counter()
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)
    report_data = report.as_dict()

    dataset_drift_result: dict[str, Any] = {}
    feature_drift_result: dict[str, Any] = {}
    for metric in report_data.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            dataset_drift_result = metric.get("result", {})
        elif metric.get("metric") == "DataDriftTable":
            feature_drift_result = metric.get("result", {})
    per_feature = feature_drift_result.get("drift_by_columns", {})
    drifted = [name for name, value in per_feature.items() if value.get("drift_detected")]
    share = float(dataset_drift_result.get(
        "share_of_drifted_columns",
        len(drifted) / max(len(feature_columns), 1),
    ))
    detected = share >= request.drift_share_threshold

    DRIFT_DETECTED.set(1 if detected else 0)
    DRIFT_SCORE.set(share)
    DRIFTED_FEATURES.set(len(drifted))
    for name in feature_columns:
        FEATURE_DRIFT.labels(feature_name=name).set(1 if name in drifted else 0)

    now = datetime.now(timezone.utc)
    filename = f"wdbc_drift_{now.strftime('%Y%m%d_%H%M%S_%f')}.html"
    report.save_html(str(REPORTS_DIR / filename))
    duration = time.perf_counter() - started
    ANALYSIS_TOTAL.inc()
    ANALYSIS_DURATION.observe(duration)
    LAST_ANALYSIS.set(now.timestamp())
    last_analysis = now.isoformat()

    return {
        "status": "success",
        "drift_detected": detected,
        "drift_share": share,
        "drifted_features": drifted,
        "feature_count": len(feature_columns),
        "reference_samples": len(reference),
        "production_samples": len(current),
        "duration_seconds": duration,
        "report": f"/reports/{filename}",
    }


@app.get("/reports")
def list_reports() -> dict[str, Any]:
    reports = sorted(REPORTS_DIR.glob("wdbc_drift_*.html"), key=lambda path: path.stat().st_mtime, reverse=True)
    return {"reports": [{"filename": path.name, "url": f"/reports/{path.name}"} for path in reports]}


@app.get("/reports/{filename}", response_class=HTMLResponse)
def get_report(filename: str) -> HTMLResponse:
    if Path(filename).name != filename or not filename.endswith(".html"):
        raise HTTPException(status_code=400, detail="invalid report name")
    path = REPORTS_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="report not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
