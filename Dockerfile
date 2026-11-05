# Dockerfile for StorySynth: Multi‑Modal Narrative Generator
# -----------------------------------------------------------
# This Dockerfile builds a production‑ready container image for the
# StorySynth application, which uses FastAPI, PyTorch, Transformers,
# Diffusers, and Jinja2 to generate multi‑modal narratives.

# ---- Base Image -------------------------------------------------
FROM python:3.11-slim-bookworm AS base

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Reduce the size of the final image by not writing .pyc files
    PIP_NO_CACHE_DIR=1 \
    # Ensure pip uses the official PyPI index
    PIP_INDEX_URL=https://pypi.org/simple

# ---- System Dependencies -----------------------------------------
# Install OS packages required for building wheels and runtime libs.
# - gcc, g++, make: needed for compiling some Python extensions.
# - libglib2.0-0, libgl1, libglib2.0-dev: required by OpenCV / image libs.
# - ffmpeg: for audio/video processing (used by diffusers pipelines).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        make \
        git \
        libglib2.0-0 \
        libgl1 \
        libglib2.0-dev \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ---- Python Dependencies -----------------------------------------
# Create a virtual environment to isolate packages.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy only the requirements file first to leverage Docker layer caching.
WORKDIR /app
COPY requirements.txt .

# Install Python packages. The torch package is installed from the
# official PyPI wheels which already contain CUDA support if needed.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---- Application Code --------------------------------------------
# Copy the rest of the application source code.
COPY . /app

# ---- Non‑Root User ------------------------------------------------
# Create a non‑root user for running the service.
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -d /app -s /sbin/nologin appuser && \
    chown -R appuser:appgroup /app

USER appuser

# ---- Expose Port -------------------------------------------------
EXPOSE 8000

# ---- Healthcheck -------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
  CMD curl -f http://localhost:8000/health || exit 1

# ---- Runtime Command ----------------------------------------------
# Use gunicorn with Uvicorn workers for production performance.
CMD ["gunicorn", "app.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--log-level", "info"]