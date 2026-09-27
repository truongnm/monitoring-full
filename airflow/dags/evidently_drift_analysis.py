"""Run a feature-drift report from recent API captures every five minutes."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

EVIDENTLY_URL = os.getenv("EVIDENTLY_URL", "http://evidently:8001")


def analyze_recent_predictions() -> None:
    response = requests.post(
        f"{EVIDENTLY_URL}/analyze",
        json={"window_size": 200, "drift_share_threshold": 0.5},
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("status") == "waiting_for_samples":
        logging.info(
            "Waiting for enough predictions for drift analysis: %s/%s",
            result.get("available"), result.get("required"),
        )
        return
    logging.info(
        "Evidently analysis finished: drift=%s, features=%s, report=%s",
        result.get("drift_detected"), result.get("drifted_features"), result.get("report"),
    )


with DAG(
    dag_id="evidently_wdbc_drift_analysis",
    description="Analyze recent WDBC predictions for feature drift",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule="*/5 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["monitoring", "evidently", "drift"],
) as dag:
    PythonOperator(
        task_id="analyze_recent_predictions",
        python_callable=analyze_recent_predictions,
        retries=2,
    )
