"""Poll Prometheus alert rules and send state changes to a Telegram chat."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
STATE_VARIABLE = "prometheus_telegram_alert_state"


def _post_telegram(token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    body = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=15) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not result.get("ok"):
        raise RuntimeError(f"Telegram API rejected the message: {result}")


def check_prometheus_alerts() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    request = Request(f"{PROMETHEUS_URL}/api/v1/alerts")
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach Prometheus at {PROMETHEUS_URL}: {exc}") from exc

    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus returned an unsuccessful alert response: {payload}")

    alerts = payload.get("data", {}).get("alerts", [])
    firing = {}
    for alert in alerts:
        if alert.get("state") != "firing":
            continue
        labels = alert.get("labels", {})
        key = hashlib.sha256(
            json.dumps(labels, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        annotations = alert.get("annotations", {})
        firing[key] = {
            "name": labels.get("alertname", "Prometheus alert"),
            "severity": labels.get("severity", "unknown"),
            "summary": annotations.get("summary", "No summary supplied"),
            "description": annotations.get("description", ""),
        }

    previous = json.loads(Variable.get(STATE_VARIABLE, default_var="{}"))
    newly_firing = {key: value for key, value in firing.items() if key not in previous}
    resolved = {key: value for key, value in previous.items() if key not in firing}

    if not token or not chat_id:
        logging.warning(
            "Telegram is not configured. TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are needed; "
            "current firing alerts: %s",
            [item["name"] for item in firing.values()],
        )
        return

    if newly_firing:
        lines = ["🚨 DDM501 monitoring alert"]
        for item in newly_firing.values():
            lines.append(f"\n[{item['severity'].upper()}] {item['name']}: {item['summary']}")
            if item["description"]:
                lines.append(item["description"])
        _post_telegram(token, chat_id, "\n".join(lines)[:4000])

    if resolved:
        names = sorted({item["name"] for item in resolved.values()})
        _post_telegram(token, chat_id, "✅ Resolved: " + ", ".join(names)[:3900])

    Variable.set(STATE_VARIABLE, json.dumps(firing))
    logging.info(
        "Prometheus reports %d firing alerts; %d new, %d resolved",
        len(firing), len(newly_firing), len(resolved),
    )


with DAG(
    dag_id="prometheus_telegram_alerts",
    description="Forward Prometheus alert changes to Telegram",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule="*/1 * * * *",
    catchup=False,
    max_active_runs=1,
    tags=["monitoring", "telegram"],
) as dag:
    PythonOperator(
        task_id="check_alerts_and_notify",
        python_callable=check_prometheus_alerts,
        retries=2,
    )
