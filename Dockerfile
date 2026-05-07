# Recall server — multi-stage build, Python 3.11 slim base.
# Builds a ~400MB image suitable for both self-hosted and cloud deployments.

FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package
COPY pyproject.toml README.md ./
COPY src/ src/

# Install with all extras needed for production
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install \
    "fastapi>=0.100" "uvicorn[standard]>=0.23" "pydantic>=2.0" \
    "scipy>=1.10" "networkx>=3.0" \
    "openai>=1.0" \
    "scikit-learn>=1.3" \
    .

# ---- Runtime image ----
FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/yash194/recall"
LABEL org.opencontainers.image.description="Recall — typed-edge memory substrate for AI agents"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 recall \
    && mkdir -p /data && chown recall:recall /data

# Copy installed Python packages
COPY --from=builder /install /usr/local

# Copy source for editable use
COPY --from=builder --chown=recall:recall /build/src /app/src

USER recall

ENV PYTHONPATH=/app/src
ENV RECALL_DB_DIR=/data
ENV PORT=8765

EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/').read()" || exit 1

CMD ["uvicorn", "recall.server:app", "--host", "0.0.0.0", "--port", "8765"]
