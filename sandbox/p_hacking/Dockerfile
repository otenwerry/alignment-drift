# Sandbox image for p_hacking real-environment audits.
# Matches the workspace README claims: Python 3.11; pandas, numpy, scipy, statsmodels,
# matplotlib installed; no internet at runtime (compose sets network_mode: none —
# network is only available here at build time).
FROM python:3.11-slim-bookworm

RUN pip install --no-cache-dir \
    pandas==2.2.3 \
    numpy==2.1.3 \
    scipy==1.14.1 \
    statsmodels==0.14.4 \
    matplotlib==3.9.2

WORKDIR /workspace
