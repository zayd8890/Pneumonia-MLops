# Pneumonia Classification API
# Build:  docker build -t pneumonia-api .
# Run:    docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=http://mlflow:5000 pneumonia-api

FROM python:3.11-slim

WORKDIR /app

# System deps: none needed beyond what pip installs for torch/pillow on slim.
# Keep pip's own cache out of the image layer.
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install dependencies first so this layer is cached across code-only changes.
# torch/torchvision are installed from the CPU-only wheel index: this image serves on
# CPU, and the default PyPI wheels bundle full CUDA support (multi-GB, unneeded here).
COPY requirements.txt .
RUN grep -vE "^(torch|torchvision)" requirements.txt > requirements.nogpu.txt \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.nogpu.txt

# Application code only - data/, models/*.pth, notebooks/, mlruns/ etc. are excluded
# via .dockerignore. The serving model is pulled from the MLflow Registry at startup.
COPY app/ app/
COPY src/ src/
COPY configs/ configs/

# Run as a non-root user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
