# DocVault — production Docker image
# No torch, no sentence-transformers, no local ML models in base image
# Embeddings: OpenAI API | Reranker + Verifier: ONNX Runtime
# Image size: ~800MB (down from ~6GB)

FROM python:3.11-slim AS builder

WORKDIR /build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# Install only production dependencies (no torch, no local extras)
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip wheel --no-cache-dir \
      --wheel-dir /build/wheels \
      ".[default]" 2>/dev/null || \
    python -m pip wheel --no-cache-dir \
      --wheel-dir /build/wheels \
      .

# Runtime stage — minimal
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install wheels from builder
COPY --from=builder /build/wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/*.whl 2>/dev/null; \
    pip install --no-cache-dir --find-links /tmp/wheels /tmp/wheels/*.whl 2>/dev/null; \
    rm -rf /tmp/wheels

# Copy application
COPY src ./src
COPY pyproject.toml .
RUN pip install --no-cache-dir --no-deps -e .

COPY corpus ./corpus
COPY eval ./eval
COPY dashboards ./dashboards

# Copy pre-exported ONNX models (reranker + verifier)
# Run `python scripts/export_onnx.py` locally first
COPY models ./models

EXPOSE 8000

# Liveness probe: process alive
# Readiness probe: models loaded + DB connected
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health/ready || exit 1

CMD ["python", "-m", "docvault.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
