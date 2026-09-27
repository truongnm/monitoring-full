# Tutorial 04 — Prometheus and Grafana

**DDM501 — AI in DevOps, DataOps, MLOps · FSB, FPT University**


## What this tutorial is for

The lab hands you finished dashboards and alert rules and has you fill in the instrumentation. Here you start from a service with **no metrics at all** and add them one at a time, asking each time what question the new metric answers.

## Setup

**With Docker** — runs the API, Prometheus, Grafana, and Airflow

```bash
docker compose up --build
```

To run the stack in the background, use `docker compose up --build -d`.

The model is trained *inside* the image build
Re-training means re-building — correct, because the model is a build artefact,
and it also guarantees the pickle matches the scikit-learn version in the image.

**Without Docker**  

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_model.py          # once
uvicorn app.main:app --port 8000
```

| | URL | Note |
|---|---|---|
| API | <http://127.0.0.1:18000/docs> | 8000 without Docker |
| Prometheus | <http://127.0.0.1:19090> | Status → Targets shows `wdbc-api` UP; compose waits for the API to be *healthy* before starting Prometheus, so it should be UP on the first look |
| Grafana | <http://127.0.0.1:13000> | anonymous viewer, dashboard `DDM501 / WDBC API` |
| Airflow | <http://127.0.0.1:18081> | local tutorial login: `admin` / `admin`; DAG `prometheus_telegram_alerts` checks alerts every minute |

## Telegram alerts

Airflow polls Prometheus's alert API every minute. It sends one Telegram message
when an alert starts firing and a resolution message when it clears. The DAG
tracks alert state in Airflow so a long-running alert does not send a message
every minute.

1. Create a Telegram bot and get its bot token; get the chat ID of the user or
   group that should receive messages.
2. Copy `.env.example` to `.env` and set `TELEGRAM_BOT_TOKEN` and
   `TELEGRAM_CHAT_ID`.
3. Apply the environment variables by recreating Airflow:

   ```bash
   docker compose up -d --force-recreate airflow-webserver airflow-scheduler
   ```

The API, Prometheus, Grafana, and Airflow UI still run if Telegram credentials
are absent; the DAG logs that Telegram notifications are not configured.
Prometheus alert rules are in `monitoring/prometheus/alerts/model.yml`.

## How the project works

The API loads the trained scikit-learn model when it starts. `/predict` checks
that all expected WDBC features are present and returns a benign/malignant
prediction. `/metrics` exposes request counters, prediction latency, model
readiness, and the recent malignant prediction share in Prometheus format.

Prometheus scrapes `/metrics` every five seconds and evaluates the alert rules.
Grafana reads Prometheus and displays the provisioned WDBC dashboard. Airflow
is the notification workflow: its scheduler runs the DAG each minute, the DAG
checks Prometheus for firing/resolved alerts, and the task calls Telegram's
`sendMessage` API when credentials are configured. PostgreSQL stores Airflow's
metadata; Docker volumes preserve its metadata, logs, and Prometheus/Grafana
data across restarts.

## Making something to look at

569 rows in a file draw no graphs. Leave this running in its own terminal:

```bash
python scripts/traffic.py --rps 20 --seconds 300
python scripts/traffic.py --broken 0.2        # 20% malformed requests
python scripts/traffic.py --drift 3.0         # shift every input by 3 sigma
```

## The exercises

| | Do this | Look for |
|---|---|---|
| 1 | `curl localhost:8000/metrics` | The whole contract with Prometheus, in plain text. Read it once. |
| 2 | Traffic, then `python scripts/count_series.py` | 23 time series. Remember the number. |
| 3 | `python scripts/traffic.py --broken 0.2` | `wdbc_errors_total{reason="missing_features"}` climbs; the error-share panel moves. |
| 4 | `python scripts/traffic.py --drift 3.0` | Latency unchanged, error rate unchanged, `wdbc_malignant_share` moves a long way. Nothing is broken and the answers changed. |
| 5 | Restart with `T04_TRAP=1`, send traffic, run `count_series.py` again | 639 series instead of 23, for the same requests. |

## Checklist

1. Why is a Counter almost never read directly, and what do you wrap it in?
2. Your average latency is 40 ms. Name two very different situations that both
   produce that number, and say which metric type tells them apart.
3. `wdbc_model_info` is a gauge permanently stuck at 1. What is it for?
4. Every alert in `monitoring/prometheus/alerts/model.yml` has a `for:` clause.
   What breaks if you remove them?
5. The trap in exercise 5 turned 23 series into 639. What was the label, and
   why is the number unbounded rather than merely large?
6. During the `--drift 3.0` run, which of the four dashboard panels moved and
   which did not? What kind of failure is that, and would a normal web-service
   alert have caught it?
