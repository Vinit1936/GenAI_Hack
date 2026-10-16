# Multi-stage production Dockerfile for FarmWatch FastAPI Backend
FROM python:3.11-slim as builder

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies using PyTorch CPU wheels for compact image size
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final minimal runtime image
FROM python:3.11-slim as runner

WORKDIR /app

# Create non-root unprivileged service user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/artifacts/models && \
    chown -R appuser:appuser /app

# Copy installed Python packages from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application source and model weights
COPY --chown=appuser:appuser backend/ /app/backend/
COPY --chown=appuser:appuser training/ /app/training/
COPY --chown=appuser:appuser artifacts/models/farmwatch_resnet18.pth /app/artifacts/models/farmwatch_resnet18.pth

ENV PATH="/home/appuser/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FARMWATCH_MODEL_PATH="/app/artifacts/models/farmwatch_resnet18.pth" \
    HOST="0.0.0.0" \
    PORT=8000

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; res = urllib.request.urlopen('http://127.0.0.1:8000/health'); exit(0 if res.getcode() == 200 else 1)"

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
