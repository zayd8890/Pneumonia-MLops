#!/bin/sh
# Builds the Postgres connection string from env vars at container start, so the
# password never has to be hardcoded in the image/compose file - set it via a
# gitignored .env file (see .env.example) or your deployment's secret manager.
set -e

exec mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri "postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}" \
  --default-artifact-root mlflow-artifacts:/ \
  --artifacts-destination /mlartifacts \
  --allowed-hosts "mlflow:5000,localhost:5000,127.0.0.1:5000,localhost:5001,127.0.0.1:5001,mlflow,localhost,127.0.0.1"
