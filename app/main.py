"""LlamaDeck — FastAPI application for managing llama-server across multiple GPUs."""

from __future__ import annotations

import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .gpu import get_gpu_status
from .llama import InstanceManager
from .models import (
    Config,
    GPUStatus,
    InstanceCreate,
    InstanceUpdate,
    LlamaInstance,
    ModelFile,
)


# ── Configuration ────────────────────────────────────────────────────

_config = Config(
    models_dir=os.environ.get("MODELS_DIR", "/models"),
    llama_server_binary=os.environ.get("LLAMA_SERVER_BINARY", "llama-server"),
    default_host=os.environ.get("DEFAULT_HOST", "0.0.0.0"),
    port_range_start=int(os.environ.get("PORT_RANGE_START", "8081")),
    port_range_end=int(os.environ.get("PORT_RANGE_END", "9000")),
    data_dir=os.environ.get("DATA_DIR", "data"),
)

_manager: InstanceManager | None = None


# ── Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager
    Path(_config.data_dir).mkdir(parents=True, exist_ok=True)
    _manager = InstanceManager(_config)
    print(f"[llama-deck] Starting — models_dir={_config.models_dir}, port range={_config.port_range_start}-{_config.port_range_end}")
    yield
    print("[llama-deck] Shutting down...")
    _manager.shutdown()


# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="LlamaDeck",
    description="Lightweight manager for llama-server across multiple GPUs",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


def _mgr() -> InstanceManager:
    assert _manager is not None
    return _manager


# ── Web UI ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    """Serve the web UI."""
    index_path = _static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>LlamaDeck</h1><p>index.html not found</p>")


# ── API: GPU ─────────────────────────────────────────────────────────

@app.get("/api/gpus", response_model=GPUStatus)
async def api_gpus():
    """Get GPU status."""
    return get_gpu_status()


# ── API: Instances ───────────────────────────────────────────────────

@app.get("/api/instances", response_model=list[LlamaInstance])
async def api_list_instances():
    """List all instances."""
    return _mgr().list_instances()


@app.get("/api/instances/{instance_id}", response_model=LlamaInstance)
async def api_get_instance(instance_id: str):
    """Get a specific instance."""
    inst = _mgr().get_instance(instance_id)
    if not inst:
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
    return inst


@app.post("/api/instances", response_model=LlamaInstance, status_code=201)
async def api_create_instance(data: InstanceCreate):
    """Create a new instance."""
    try:
        return _mgr().create_instance(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/instances/{instance_id}", response_model=LlamaInstance)
async def api_update_instance(instance_id: str, data: InstanceUpdate):
    """Update an instance (must be stopped)."""
    try:
        inst = _mgr().update_instance(instance_id, data)
        if not inst:
            raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
        return inst
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.delete("/api/instances/{instance_id}")
async def api_delete_instance(instance_id: str):
    """Delete an instance (stops it first if running)."""
    if not _mgr().delete_instance(instance_id):
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
    return {"ok": True}


# ── API: Start / Stop ───────────────────────────────────────────────

@app.post("/api/instances/{instance_id}/start", response_model=LlamaInstance)
async def api_start_instance(instance_id: str):
    """Start a llama-server instance."""
    try:
        return _mgr().start_instance(instance_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/instances/{instance_id}/stop", response_model=LlamaInstance)
async def api_stop_instance(instance_id: str):
    """Stop a llama-server instance."""
    try:
        return _mgr().stop_instance(instance_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")


@app.post("/api/instances/start-all")
async def api_start_all():
    """Start all stopped instances."""
    return _mgr().start_all()


@app.post("/api/instances/stop-all")
async def api_stop_all():
    """Stop all running instances."""
    return _mgr().stop_all()


# ── API: Logs ────────────────────────────────────────────────────────

@app.get("/api/instances/{instance_id}/logs")
async def api_get_logs(instance_id: str, lines: int = Query(default=200, le=2000)):
    """Get recent logs for an instance."""
    inst = _mgr().get_instance(instance_id)
    if not inst:
        raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
    return {"logs": _mgr().get_logs(instance_id, lines)}


# ── API: Models ──────────────────────────────────────────────────────

@app.get("/api/models", response_model=list[ModelFile])
async def api_list_models():
    """List available model files (.gguf) in the models directory."""
    return _mgr().list_models()


# ── API: Overall status ──────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    """Get overall system status — GPUs, instances, models, config."""
    return {
        "gpu": get_gpu_status().model_dump(),
        "instances": [i.model_dump() for i in _mgr().list_instances()],
        "models": [m.model_dump() for m in _mgr().list_models()],
        "config": _config.model_dump(),
    }


# ── Run ──────────────────────────────────────────────────────────────

def main():
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))

    print(f"[llama-deck] Listening on {host}:{port}")
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=os.environ.get("RELOAD", "0") == "1",
        log_level="info",
    )


if __name__ == "__main__":
    main()
