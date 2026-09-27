# Tutorial 04 — Prometheus and Grafana

**DDM501 — AI in DevOps, DataOps, MLOps · FSB, FPT University**


## What this tutorial is for

The lab hands you finished dashboards and alert rules and has you fill in the instrumentation. Here you start from a service with **no metrics at all** and add them one at a time, asking each time what question the new metric answers.

## Setup

**With Docker** — runs the API, MLflow/Model Registry, Silo (MinIO-compatible S3 storage), Evidently, Prometheus,
Grafana, and Airflow

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
| MLflow | <http://127.0.0.1:15000> | experiment tracking and WDBC model registry |
| S3 Console (Silo) | <http://127.0.0.1:19001> | local object storage; `minio` / `minio123` by default |
| Evidently | <http://127.0.0.1:18001/docs> | capture predictions, run drift reports, browse `/reports` |
| Prometheus | <http://127.0.0.1:19090> | Status → Targets shows `wdbc-api` UP; compose waits for the API to be *healthy* before starting Prometheus, so it should be UP on the first look |
| Grafana | <http://127.0.0.1:13000> | anonymous viewer, dashboard `DDM501 / WDBC API` |
| Airflow | <http://127.0.0.1:18081> | local tutorial login: `admin` / `admin`; DAG `prometheus_telegram_alerts` checks alerts every minute |

The API model registry details are available at
<http://127.0.0.1:18000/model/info>. The local no-Docker instructions run the
API from a joblib file; the full MLflow, S3-compatible storage, and Evidently stack is enabled
through Docker Compose.

## Telegram alerts

Airflow polls Prometheus's alert API every minute. It sends one Telegram message
when an alert starts firing and a resolution message when it clears. The DAG
tracks alert state in Airflow so a long-running alert does not send a message
every minute.

1. In Telegram, open `@BotFather`, send `/newbot`, follow its prompts, and save
   the bot token privately. Telegram's [bot creation guide](https://core.telegram.org/bots/features#creating-a-new-bot)
   explains this step. Never commit or share the token.
2. Open the new bot's chat and send `/start` so it can message you. To find your
   private chat ID, query the Bot API `getUpdates` method after sending that
   message, then read `message.chat.id` from the returned update. The official
   [Bot API reference](https://core.telegram.org/bots/api#getupdates) describes
   the response. For a group, add the bot to the group, send a message there,
   then use that group's `chat.id`.
3. Copy `.env.example` to `.env` in the project root and set
   `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`. In PowerShell:

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   Keep the `.env` file private; it is ignored by Git.
4. Apply the environment variables by recreating Airflow:

   ```bash
   docker compose up -d --force-recreate airflow-webserver airflow-scheduler
   ```

The API, Prometheus, Grafana, and Airflow UI still run if Telegram credentials
are absent; the DAG logs that Telegram notifications are not configured.
Prometheus alert rules are in `monitoring/prometheus/alerts/model.yml`.
The bot only sends when a Prometheus alert changes to `firing` or resolves; it
does not send a startup/test message. For a controlled end-to-end check, let
`ApiDown` fire by stopping the API for over one minute, then start it again and
wait for Prometheus to clear the alert. Airflow's DAG task logs show Telegram
API errors if the token or chat ID is invalid.

## How the project works

The API loads the trained scikit-learn model when it starts. `/predict` checks
that all expected WDBC features are present and returns a benign/malignant
prediction. `/metrics` exposes request counters, prediction latency, model
readiness, and the recent malignant prediction share in Prometheus format.

During the Docker image build, the WDBC model is trained using a stratified,
fixed-seed 80/20 split and the pinned scikit-learn version. The training step
records accuracy, precision, recall, F1, ROC AUC, and a labeled confusion matrix
in `models/model_card.json`. The registration service recomputes these metrics
on the same holdout split, logs them and the model card to MLflow, and only
promotes a model whose ROC AUC meets `MIN_TEST_ROC_AUC` (0.90 by default). It
stores the model artifact in S3-compatible Silo storage. The API loads the registered Production
model and exposes `POST /model/reload` to load a newly promoted version without
restarting the container. Without Docker, the original local `joblib` model
path remains available.

Each successful prediction is also captured by Evidently, which compares recent
input features with the WDBC reference data and writes HTML reports. Airflow
runs that analysis every five minutes; Prometheus scrapes its drift metrics,
and Grafana's `WDBC Feature Drift - Evidently` dashboard shows drift by feature.
The existing Airflow Telegram DAG checks Prometheus firing/resolved alerts each
minute. PostgreSQL stores Airflow and MLflow metadata; Silo stores model
artifacts; Docker volumes preserve metadata, reports, and monitoring data.

## Making something to look at

569 rows in a file draw no graphs. Leave this running in its own terminal:

**With Docker Compose**, run the traffic generator inside the API container. It
has the trained model card and Python dependencies already:

```bash
docker compose exec api python scripts/traffic.py --rps 20 --seconds 300
docker compose exec api python scripts/traffic.py --broken 0.2
docker compose exec api python scripts/traffic.py --drift 3.0 --analyze
docker compose exec api python scripts/count_series.py
```

Evidently runs a report on demand with `--analyze`; otherwise Airflow runs it
every five minutes after 200 predictions have been captured. Open
<http://127.0.0.1:18001/reports> to see the generated HTML reports.

**Without Docker**, activate the virtual environment from Setup and run:

```bash
python scripts/traffic.py --rps 20 --seconds 300
python scripts/traffic.py --broken 0.2        # 20% malformed requests
python scripts/traffic.py --drift 3.0         # shift every input by 3 sigma
python scripts/count_series.py
```

## The exercises

| | Do this | Look for |
|---|---|---|
| 1 | Docker: `curl localhost:18000/metrics`; without Docker: `curl localhost:8000/metrics` | The whole contract with Prometheus, in plain text. Read it once. |
| 2 | Send normal traffic, then run `count_series.py` using the command for your setup above | Observe the number of WDBC time series and remember it. |
| 3 | Run `traffic.py --broken 0.2` using the command for your setup above | `wdbc_errors_total{reason="missing_features"}` climbs; the error-share panel moves. |
| 4 | Run `traffic.py --drift 3.0 --analyze` using the command for your setup above | Latency and error rate stay steady while malignant share and Evidently feature drift can move. |
| 5 | Set `T04_TRAP=1` in `.env`, run `docker compose up -d --force-recreate api`, send traffic, then run `count_series.py` again | The `sample_id` label creates a new series for each patient. Restore `T04_TRAP=0` and recreate the API when finished. |

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
