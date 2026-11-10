# syntax=docker/dockerfile:1

FROM python:3.11-slim

ARG INSTALL_ML=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WASTECAP_BACKEND=mock \
    WASTECAP_DEVICE=cpu \
    WASTECAP_QUANTIZE=none \
    WASTECAP_DB_PATH=/workspace/data/db/wastecap.db \
    WASTECAP_CONFIGS_DIR=/workspace/configs \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /workspace

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY scripts ./scripts
COPY .streamlit ./.streamlit

RUN python -m pip install --upgrade pip \
    && if [ "$INSTALL_ML" = "true" ]; then \
         python -m pip install ".[ml]"; \
       else \
         python -m pip install .; \
       fi \
    && useradd --create-home --shell /usr/sbin/nologin wastecap \
    && mkdir -p /workspace/data/db /workspace/data/processed /workspace/results \
    && chown -R wastecap:wastecap /workspace

USER wastecap

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://127.0.0.1:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/main.py"]
