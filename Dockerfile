# Dockerfile for KDA-CSA-HCA Fusion reference implementation
# Python 3.11 slim base for reproducibility and small image size
FROM python:3.11-slim

# Metadata
LABEL maintainer="kda-csa-hca-fusion contributors"
LABEL description="Containerized PyTorch linear-attention hybrid architecture reference implementation"

# Prevent python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies for building wheels and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /workspace

# Copy dependency manifests first for better layer caching
COPY requirements.txt pyproject.toml ./
# Optionally copy constraints if present
RUN pip install --upgrade pip && \
    pip install -e ".[dev]" || pip install -r requirements.txt

# Copy source code
COPY . .

# Create backup directory
RUN mkdir -p /workspace/backups

# Default command runs CI gate (syntax check)
CMD ["python3", "ci_gate.py", "--syntax-only"]
