FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

# Install Python 3.12 + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    curl wget libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app/ /app/app/

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
