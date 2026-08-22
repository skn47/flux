# Packages the API server only -- the ingestion/labeling/lstm/backtest
# pipeline keeps running via the host's own .venv + crontab (see
# scripts/refresh_pipeline.sh, scripts/refresh_backtests.sh, and
# api/README.md's "Data freshness"/"Deployment" sections). This image never
# runs that pipeline; docker-compose.yml volume-mounts the host's data/,
# lstm/models/, and backtest/ directories straight into the container so
# every cron-driven refresh is visible immediately, no rebuild required.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
