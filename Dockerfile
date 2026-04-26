# ─────────────────────────────────────────────────────────
#  NexusMD — Railway Deployment Dockerfile
#  Single container: FastAPI backend + static frontend
#  Port: $PORT (Railway injects this automatically)
# ─────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="NexusMD"
LABEL description="NexusMD Drug Discovery Platform — Railway deployment"

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    openbabel \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /nexusmd

# Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install AutoDock Vina (Linux x86_64 — matches Railway containers)
RUN wget -q https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64 \
    -O /usr/local/bin/vina \
    && chmod +x /usr/local/bin/vina \
    && echo "Vina installed: $(/usr/local/bin/vina --version 2>&1 | head -1)" \
    || echo "WARNING: Vina download failed — docking will use simulation mode"

# App code
COPY app/ ./app/

# Static frontend — served by FastAPI at /
COPY frontend/ ./frontend/

# Data dirs (Railway filesystem is ephemeral — results stored in memory/tmp)
RUN mkdir -p data/results data/pdb_cache data/ligands logs

# Environment
ENV VINA_BINARY=/usr/local/bin/vina
ENV OBABEL_BINARY=/usr/bin/obabel
ENV PYTHONPATH=/nexusmd
# Railway sets $PORT — default 8000 if not set
ENV PORT=8000

EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Use shell form so $PORT is expanded at runtime
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
