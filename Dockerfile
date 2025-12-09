FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/model_out /app/logs

COPY src ./src
COPY dataset.csv ./dataset.csv

ENTRYPOINT ["python", "-m", "src.cli"]
