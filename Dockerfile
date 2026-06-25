FROM python:3.12-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget \
    && rm -rf /var/lib/apt/lists/*

# Install llama.cpp server binary
# Options:
#   1. Pre-built binary (copy from host or download)
#   2. Build from source (heavier but guaranteed)
# By default we expect the binary on the PATH or mounted via volume.
# To build from source, uncomment the build section below.
#
# ── Option A: Pre-built binary (default) ──
# The Dockerfile expects llama-server to be available at build time or
# mounted at runtime. Copy it in from the host or a builder stage:
#   COPY --from=llama /build/bin/llama-server /usr/local/bin/llama-server
#
# ── Option B: Build from source (uncomment below) ──
# FROM ubuntu:24.04 AS llama-builder
# RUN apt-get update && apt-get install -y build-essential cmake git
# RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /src && \
#     cd /src && cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF && \
#     cmake --build build --config Release -j$(nproc) && \
#     cp build/bin/llama-server /usr/local/bin/llama-server
# FROM python:3.12-slim
# COPY --from=llama-builder /usr/local/bin/llama-server /usr/local/bin/llama-server

# Install Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app/ /app/app/
COPY data/ /app/data/ 2>/dev/null || true

# Create directories
RUN mkdir -p /app/data /models

# Environment
ENV MODELS_DIR=/models \
    DATA_DIR=/app/data \
    LLAMA_SERVER_BINARY=llama-server \
    PORT=8080 \
    HOST=0.0.0.0

EXPOSE 8080

CMD ["python", "-m", "app.main"]
