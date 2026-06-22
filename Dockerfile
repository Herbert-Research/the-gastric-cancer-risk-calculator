# =============================================================================
# Multi-Stage Dockerfile for Gastric Cancer Risk Calculator
# =============================================================================
# Stage 1: Builder - Install dependencies in a virtual environment
# Stage 2: Runtime - Minimal image with only runtime requirements
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Install build dependencies for scientific Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment and install dependencies
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Runtime (Production)
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Matplotlib configuration for headless operation
    MPLBACKEND=Agg

WORKDIR /app

# Install only runtime dependencies (libopenblas for numpy/scipy)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libopenblas0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY models/ ./models/
COPY utils/ ./utils/
COPY data/ ./data/
COPY risk_calculator.py .
COPY app.py .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app \
    && mkdir -p /app/.matplotlib \
    && chown -R appuser:appuser /app/.matplotlib

USER appuser

# Set matplotlib config directory to writable location
ENV MPLCONFIGDIR=/app/.matplotlib

# Default command: run the risk calculator
CMD ["python", "risk_calculator.py", "--data", "data/tcga_2018_clinical_data.tsv"]

# -----------------------------------------------------------------------------
# Stage 3: Demo (Streamlit Interactive App)
# -----------------------------------------------------------------------------
FROM runtime AS demo

# Switch back to root to install additional packages
USER root

# Install streamlit and dependencies
RUN /opt/venv/bin/pip install --no-cache-dir streamlit>=1.28 watchdog>=3.0

# Switch back to non-root user
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
