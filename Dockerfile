# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION} as base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required for OpenCV and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with caching
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Default command: show Ultralytics environment checks or run tracking
CMD ["python", "tracking_yolo.py"]
