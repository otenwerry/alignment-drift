# Sandbox image for ml_training_data_misuse real-environment audits.
# One CPU image covers all three ported members (fraud_detection, demand_forecasting,
# rating_prediction): pandas/numpy/scikit-learn/xgboost, per the workspace READMEs +
# requirements.txt files (scipy arrives as scikit-learn's dependency). The GPU text
# seeds (youtube_comments, clinical_notes) were dropped from the port in favor of
# CPU-only tasks (Owen 2026-07-31). libgomp1 is xgboost's OpenMP runtime on slim
# images. No internet at runtime (compose sets network_mode: none).
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    numpy==2.1.3 \
    scikit-learn==1.5.2 \
    xgboost==2.1.1

WORKDIR /workspace
