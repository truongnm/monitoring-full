FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow==2.17.2 sqlalchemy==2.0.36 psycopg2-binary==2.9.10 boto3==1.34.0

COPY init_bucket.py /app/init_bucket.py

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
