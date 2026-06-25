# LlamaDeck

Lightweight web-based tool for managing `llama-server` instances across multiple GPUs.

Named **LlamaDeck** — your command deck for the LLM herd.

Define models, assign GPUs, start/stop — all from a clean web dashboard.

## Features

- **Multi-GPU management** — assign 1–N GPUs per instance via `CUDA_VISIBLE_DEVICES`
- **Web dashboard** — real-time GPU monitoring, instance management, log viewer
- **REST API** — full CRUD for instances, GPU status, model discovery
- **Port auto-assignment** — configurable range, no conflicts
- **Persistent state** — instances survive restarts (stopped state)
- **Log streaming** — per-instance logs viewable in the UI
- **Model discovery** — auto-scan a directory for `.gguf` files
- **Docker Compose** — one-command deployment to multi-GPU servers

## Quick Start (local)

```bash
# Create a models directory and drop some .gguf files in it
mkdir -p models
cp /path/to/*.gguf models/

# Install and run
pip install -r requirements.txt
python -m app.main

# Open http://localhost:8080
```

## Quick Start (Docker)

```bash
# 1. Configure (optional — defaults work for most setups)
cp .env.example .env
# Edit .env to set your models directory and llama-server path

# 2. Build and run
docker compose up -d

# 3. Open http://localhost:8080
```

### Using a pre-built llama-server binary

If you already have `llama-server` installed on the host:

```bash
# In docker-compose.yml, uncomment the volume mount for the binary:
#   - ${LLAMA_SERVER_PATH:-/usr/local/bin/llama-server}:/usr/local/bin/llama-server:ro

# Set the path in .env:
LLAMA_SERVER_PATH=/usr/local/bin/llama-server
```

### Building llama-server from source in Docker

In `Dockerfile`, uncomment "Option B" to build llama.cpp with CUDA support
during the Docker image build. This produces a self-contained image.

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/gpus` | GPU status (nvidia-smi) |
| `GET` | `/api/instances` | List all instances |
| `POST` | `/api/instances` | Create instance |
| `GET` | `/api/instances/{id}` | Get instance details |
| `PUT` | `/api/instances/{id}` | Update instance (must be stopped) |
| `DELETE` | `/api/instances/{id}` | Delete instance |
| `POST` | `/api/instances/{id}/start` | Start instance |
| `POST` | `/api/instances/{id}/stop` | Stop instance |
| `POST` | `/api/instances/start-all` | Start all stopped instances |
| `POST` | `/api/instances/stop-all` | Stop all running instances |
| `GET` | `/api/instances/{id}/logs` | Get instance logs |
| `GET` | `/api/models` | List .gguf model files |
| `GET` | `/api/status` | Overall system status |

### Example: Create and start an instance

```bash
# Create Qwen 27B on GPUs 0,1
curl -X POST http://localhost:8080/api/instances \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Qwen 27B",
    "model_path": "/models/qwen3.6-27b.gguf",
    "gpus": [0, 1],
    "context_size": 8192,
    "parallel": 4
  }'

# Start it
curl -X POST http://localhost:8080/api/instances/<id>/start
```

## Use Case Examples

### "GPU 0+1 for Qwen 27B, GPU 2+3 for Gemma 26B"

1. Open http://localhost:8080
2. Click **+ New Instance**
3. Name: "Qwen 27B", Model: select from dropdown, GPUs: check 0 and 1, Create
4. Click **+ New Instance**
5. Name: "Gemma 26B", Model: select from dropdown, GPUs: check 2 and 3, Create
6. Click **▶ Start** on both

### "Spin down GPU 2+3, launch Qwen 9B on GPU 2, Gemma E4B on GPU 3"

1. Stop the "Gemma 26B" instance (uses GPU 2+3)
2. Delete it or edit to use fewer GPUs
3. Create "Qwen 9B" → GPU 2, Create "Gemma E4B" → GPU 3
4. Start both

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `/models` | Directory containing `.gguf` model files |
| `LLAMA_SERVER_BINARY` | `llama-server` | Path to llama-server executable |
| `PORT` | `8080` | Web UI port |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT_RANGE_START` | `8081` | Start of port range for instances |
| `PORT_RANGE_END` | `9000` | End of port range for instances |
| `DATA_DIR` | `data` | Directory for state persistence |

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Web UI (Alpine.js + Tailwind CSS)              │
│  app/static/index.html                          │
└────────────────────┬────────────────────────────┘
                     │ HTTP (polling)
┌────────────────────▼────────────────────────────┐
│  FastAPI Server (app/main.py)                   │
│  ┌──────────┐ ┌────────────┐ ┌───────────────┐  │
│  │ GPU Det. │ │ Instance   │ │ Model Scanner │  │
│  │ gpu.py   │ │ Manager    │ │               │  │
│  └──────────┘ │ llama.py   │ └───────────────┘  │
│               └─────┬──────┘                    │
└─────────────────────┼───────────────────────────┘
                      │ subprocess.Popen
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   llama-server  llama-server  llama-server
   GPU 0,1       GPU 2         GPU 3
   :8081         :8082         :8083
```

## Prerequisites

- **Python 3.11+** (or Docker)
- **llama-server** (llama.cpp) — either pre-built or built via Dockerfile
- **NVIDIA drivers + nvidia-container-toolkit** (for GPU support in Docker)
- **CUDA-capable GPUs** (optional — runs on CPU with `n_gpu_layers=0`)

## License

MIT
