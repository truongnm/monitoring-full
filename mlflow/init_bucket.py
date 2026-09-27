"""Create MLflow's S3 artifact bucket using the configured S3 endpoint."""

import os
import time

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError


def main() -> None:
    bucket = os.getenv("MLFLOW_ARTIFACT_BUCKET", "mlflow-artifacts")
    client = boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
        region_name="us-east-1",
    )

    for attempt in range(30):
        try:
            client.create_bucket(Bucket=bucket)
            print(f"Created S3 bucket: {bucket}")
            return
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                print(f"S3 bucket already exists: {bucket}")
                return
            raise
        except EndpointConnectionError:
            if attempt == 29:
                raise
            time.sleep(2)


if __name__ == "__main__":
    main()
