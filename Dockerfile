FROM python:3.12.12 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN python -m venv .venv

# Install CPU-only PyTorch first (~200MB vs ~2GB for CUDA version)
RUN .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt

# Remove opencv-python (pulled by ultralytics), keep only headless version
RUN .venv/bin/pip uninstall -y opencv-python opencv-python-headless \
    && .venv/bin/pip install --no-deps opencv-python-headless==4.12.0.88

# Strip unused torch/ultralytics transitive deps and build artifacts
RUN .venv/bin/pip uninstall -y scipy matplotlib networkx fontTools polars \
    && rm -rf \
    .venv/lib/python3.12/site-packages/torch/test \
    .venv/lib/python3.12/site-packages/torch/include \
    .venv/lib/python3.12/site-packages/pip \
    .venv/lib/python3.12/site-packages/setuptools

FROM python:3.12.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libxcb1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv .venv/
COPY . .

EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
