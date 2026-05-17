FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# System deps for psycopg, scientific stack, OR-Tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY eeie ./eeie
COPY scripts ./scripts
COPY alembic.ini ./alembic.ini

RUN pip install --upgrade pip && pip install . && \
    command -v eeie-ingest >/dev/null && \
    command -v eeie-train >/dev/null && \
    command -v eeie-evaluate >/dev/null

EXPOSE 8000

# Serve FastAPI by default.
CMD ["uvicorn", "eeie.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
