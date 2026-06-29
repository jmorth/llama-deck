"""llama-server process management."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import psutil

from .gpu import get_gpu_status
from .models import (
    Config,
    InstanceCreate,
    InstanceStatus,
    InstanceUpdate,
    LlamaInstance,
    ModelFile,
)


class InstanceManager:
    """Manages llama-server processes — start, stop, monitor, and persist state."""

    def __init__(self, config: Config):
        self.config = config
        self.instances: Dict[str, LlamaInstance] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._log_buffers: Dict[str, deque] = {}  # id -> deque of log lines
        self._log_threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self._state_file = Path(config.data_dir) / "state.json"
        self._log_dir = Path(config.data_dir) / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()
        self._recover_running()

    # ── Persistence ──────────────────────────────────────────────────

    def _load_state(self) -> None:
        """Load instance configs from disk."""
        if not self._state_file.exists():
            return
        try:
            with open(self._state_file, "r") as f:
                data = json.load(f)
            for inst_data in data.get("instances", []):
                inst = LlamaInstance(**inst_data)
                # Reset status — we'll re-check in _recover_running
                if inst.status == InstanceStatus.RUNNING:
                    inst.status = InstanceStatus.STOPPED
                    inst.pid = None
                self.instances[inst.id] = inst
        except Exception as e:
            print(f"[llama-deck] Warning: failed to load state: {e}")

    def _save_state(self) -> None:
        """Persist instance configs to disk."""
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "instances": [inst.model_dump() for inst in self.instances.values()]
        }
        with open(self._state_file, "w") as f:
            json.dump(data, f, indent=2)

    def _recover_running(self) -> None:
        """Check if any stored instances are actually still running."""
        for inst in list(self.instances.values()):
            if inst.status == InstanceStatus.STOPPED and inst.pid:
                if psutil.pid_exists(inst.pid):
                    try:
                        proc = psutil.Process(inst.pid)
                        if proc.is_running() and "llama" in " ".join(proc.cmdline()):
                            inst.status = InstanceStatus.RUNNING
                            self._attach_log_stream(inst.id, inst.pid)
                            continue
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                # Process is dead
                inst.pid = None
                inst.status = InstanceStatus.STOPPED
        self._save_state()

    # ── GPU availability ───────────────────────────────────────────

    def _gpus_in_use(self) -> Set[int]:
        """Return the set of GPU indices currently assigned to running instances."""
        used: Set[int] = set()
        for inst in self.instances.values():
            if inst.status == InstanceStatus.RUNNING and inst.gpus:
                used.update(inst.gpus)
        return used

    def get_gpu_availability(self) -> dict:
        """Return GPU availability info for the frontend.

        Returns dict with:
          total_gpus: total GPU count on the system
          gpus_in_use: list of GPU indices currently in use
          gpus_free: list of GPU indices that are free
          total_free: count of free GPUs
        """
        gpu_status = get_gpu_status()
        all_indices = {gpu.index for gpu in gpu_status.gpus}
        in_use = self._gpus_in_use()
        free = sorted(all_indices - in_use)
        return {
            "total_gpus": len(all_indices),
            "gpus_in_use": sorted(in_use & all_indices),
            "gpus_free": free,
            "total_free": len(free),
        }

    def can_start(self, instance_id: str) -> dict:
        """Check whether an instance can be started.

        Returns dict with:
          ok: bool
          reason: str (if not ok)
        """
        inst = self.instances.get(instance_id)
        if not inst:
            return {"ok": False, "reason": "Instance not found"}
        if inst.status == InstanceStatus.RUNNING:
            return {"ok": False, "reason": "Already running"}
        if inst.status in (InstanceStatus.STARTING, InstanceStatus.STOPPING):
            return {"ok": False, "reason": f"Instance is {inst.status.value}"}

        avail = self.get_gpu_availability()

        # Pinned GPUs — all requested must be free
        if inst.gpus:
            missing = [g for g in inst.gpus if g in avail["gpus_in_use"]]
            if missing:
                return {
                    "ok": False,
                    "reason": f"Pinned GPU(s) {missing} in use by another instance",
                }
            # Also check GPUs exist on the system
            all_system = set(avail["gpus_in_use"]) | set(avail["gpus_free"])
            missing_sys = [g for g in inst.gpus if g not in all_system]
            if missing_sys:
                return {
                    "ok": False,
                    "reason": f"GPU(s) {missing_sys} not found on system",
                }
            return {"ok": True, "reason": ""}

        # Auto-assign count — need enough free
        if inst.gpu_count is not None and inst.gpu_count > 0:
            if avail["total_free"] < inst.gpu_count:
                return {
                    "ok": False,
                    "reason": f"Need {inst.gpu_count} GPU(s), only {avail['total_free']} free",
                }
            return {"ok": True, "reason": ""}

        # No GPUs requested — CPU only, always OK
        return {"ok": True, "reason": ""}

    def _assign_gpus(self, inst: LlamaInstance) -> None:
        """Auto-assign GPUs for an instance with gpu_count set.

        Picks the first N free GPU indices and writes them into inst.gpus.
        Raises RuntimeError if not enough GPUs are available.
        """
        if not inst.gpu_count or inst.gpu_count <= 0:
            return
        avail = self.get_gpu_availability()
        if avail["total_free"] < inst.gpu_count:
            raise RuntimeError(
                f"Need {inst.gpu_count} GPU(s), only {avail['total_free']} free"
            )
        inst.gpus = avail["gpus_free"][: inst.gpu_count]

    # ── Port allocation ──────────────────────────────────────────────

    def _next_port(self) -> int:
        """Find the next available port in the configured range."""
        used_ports = {inst.port for inst in self.instances.values()}
        for port in range(self.config.port_range_start, self.config.port_range_end):
            if port not in used_ports:
                return port
        raise RuntimeError(
            f"No available ports in range {self.config.port_range_start}-{self.config.port_range_end}"
        )

    # ── CRUD ─────────────────────────────────────────────────────────

    def list_instances(self) -> List[LlamaInstance]:
        return list(self.instances.values())

    def get_instance(self, instance_id: str) -> Optional[LlamaInstance]:
        return self.instances.get(instance_id)

    def create_instance(self, data: InstanceCreate) -> LlamaInstance:
        port = data.port or self._next_port()
        inst = LlamaInstance(
            name=data.name,
            model_path=data.model_path,
            gpus=data.gpus,
            gpu_count=data.gpu_count,
            port=port,
            host=data.host,
            context_size=data.context_size,
            n_gpu_layers=data.n_gpu_layers,
            parallel=data.parallel,
            flash_attention=data.flash_attention,
            additional_args=data.additional_args,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self.instances[inst.id] = inst
            self._save_state()
        return inst

    def update_instance(self, instance_id: str, data: InstanceUpdate) -> Optional[LlamaInstance]:
        inst = self.instances.get(instance_id)
        if not inst:
            return None
        if inst.status == InstanceStatus.RUNNING:
            raise RuntimeError("Cannot update a running instance — stop it first")
        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(inst, key, value)
        with self._lock:
            self._save_state()
        return inst

    def delete_instance(self, instance_id: str) -> bool:
        inst = self.instances.get(instance_id)
        if not inst:
            return False
        if inst.status == InstanceStatus.RUNNING:
            self._kill_process(instance_id)
        with self._lock:
            del self.instances[instance_id]
            self._log_buffers.pop(instance_id, None)
            self._save_state()
        return True

    # ── Start / Stop ─────────────────────────────────────────────────

    def start_instance(self, instance_id: str) -> LlamaInstance:
        inst = self.instances.get(instance_id)
        if not inst:
            raise KeyError(f"Instance {instance_id} not found")
        if inst.status == InstanceStatus.RUNNING:
            raise RuntimeError(f"Instance {instance_id} is already running")

        inst.status = InstanceStatus.STARTING
        inst.error_message = None

        # Auto-assign GPUs if gpu_count is set
        if inst.gpu_count and not inst.gpus:
            self._assign_gpus(inst)
            self._save_state()  # persist the assigned GPUs

        # Build command
        cmd = self._build_command(inst)
        env = os.environ.copy()

        # Set CUDA_VISIBLE_DEVICES for multi-GPU
        if inst.gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in inst.gpus)

        try:
            # Open log file
            log_file = self._log_dir / f"{instance_id}.log"
            log_fh = open(log_file, "a")

            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid,  # new process group for clean kill
            )

            inst.pid = proc.pid
            inst.status = InstanceStatus.RUNNING
            inst.started_at = datetime.now(timezone.utc).isoformat()

            with self._lock:
                self._processes[instance_id] = proc
                self._save_state()

            # Start log reader thread
            self._attach_log_stream(instance_id, proc.pid)

            # Check if process died immediately
            time.sleep(0.5)
            if proc.poll() is not None:
                inst.status = InstanceStatus.ERROR
                inst.error_message = f"Process exited immediately with code {proc.returncode}"
                inst.pid = None
                with self._lock:
                    self._save_state()

        except Exception as e:
            inst.status = InstanceStatus.ERROR
            inst.error_message = str(e)
            inst.pid = None
            with self._lock:
                self._save_state()

        return inst

    def stop_instance(self, instance_id: str) -> LlamaInstance:
        inst = self.instances.get(instance_id)
        if not inst:
            raise KeyError(f"Instance {instance_id} not found")
        if inst.status != InstanceStatus.RUNNING:
            inst.status = InstanceStatus.STOPPED
            inst.pid = None
            with self._lock:
                self._save_state()
            return inst

        inst.status = InstanceStatus.STOPPING
        self._kill_process(instance_id)
        inst.status = InstanceStatus.STOPPED
        inst.pid = None
        inst.started_at = None
        # Clear auto-assigned GPUs so next start picks fresh ones
        if inst.gpu_count:
            inst.gpus = []
        with self._lock:
            self._save_state()
        return inst

    def stop_all(self) -> List[LlamaInstance]:
        results = []
        for inst_id in list(self.instances.keys()):
            try:
                results.append(self.stop_instance(inst_id))
            except Exception:
                pass
        return results

    def start_all(self) -> List[LlamaInstance]:
        results = []
        for inst_id in list(self.instances.keys()):
            inst = self.instances.get(inst_id)
            if inst and inst.status == InstanceStatus.STOPPED:
                try:
                    results.append(self.start_instance(inst_id))
                except Exception:
                    pass
        return results

    # ── Logs ─────────────────────────────────────────────────────────

    def get_logs(self, instance_id: str, lines: int = 200) -> List[str]:
        """Get recent log lines for an instance."""
        buf = self._log_buffers.get(instance_id)
        if buf:
            return list(buf)[-lines:]

        # Fallback: read from log file
        log_file = self._log_dir / f"{instance_id}.log"
        if log_file.exists():
            with open(log_file, "r") as f:
                all_lines = f.readlines()
            return [l.rstrip() for l in all_lines[-lines:]]
        return []

    # ── Model discovery ──────────────────────────────────────────────

    def list_models(self) -> List[ModelFile]:
        """List .gguf files in the models directory.

        Tags are derived from the folder layout relative to models_dir:
          /models/<author>/<model>/<file>.gguf  →  author="Google", model_tag="Gemma 4"
        Kebab-case segments are title-cased; a leading 'v' followed by digits is kept as-is.
        """
        models_dir = Path(self.config.models_dir)
        if not models_dir.exists():
            return []

        models = []
        for f in sorted(models_dir.rglob("*.gguf")):
            # Skip multimodal projector files — not usable as standalone models
            if "mmproj" in f.name.lower():
                continue
            # Skip MTP draft models — not usable as standalone models
            if f.name.lower().startswith("mtp-"):
                continue
            size = f.stat().st_size

            # Derive tags from relative folder path
            rel = f.relative_to(models_dir)
            parts = rel.parts  # e.g. ("google", "gemma-4", "model.gguf")
            author = None
            model_tag = None
            if len(parts) >= 3:
                author = _title_tag(parts[0])
                model_tag = _title_tag(parts[1])
            elif len(parts) == 2:
                # /models/<model>/<file>.gguf — treat as model tag only
                model_tag = _title_tag(parts[0])

            models.append(
                ModelFile(
                    name=f.name,
                    path=str(f),
                    size_bytes=size,
                    size_human=_human_size(size),
                    author=author,
                    model_tag=model_tag,
                )
            )
        return models

    # ── Internal helpers ─────────────────────────────────────────────

    def _build_command(self, inst: LlamaInstance) -> List[str]:
        cmd = [self.config.llama_server_binary]
        cmd.extend(["--model", inst.model_path])
        cmd.extend(["--host", inst.host])
        cmd.extend(["--port", str(inst.port)])
        cmd.extend(["--ctx-size", str(inst.context_size)])
        cmd.extend(["--n-gpu-layers", str(inst.n_gpu_layers)])
        cmd.extend(["--parallel", str(inst.parallel)])
        if inst.flash_attention:
            cmd.extend(["--flash-attn", "on"])
        if inst.alias:
            cmd.extend(["--alias", inst.alias])
        if inst.cache_type_k:
            cmd.extend(["--cache-type-k", inst.cache_type_k])
        if inst.cache_type_v:
            cmd.extend(["--cache-type-v", inst.cache_type_v])
        if inst.temp is not None:
            cmd.extend(["--temp", str(inst.temp)])
        if inst.top_p is not None:
            cmd.extend(["--top-p", str(inst.top_p)])
        if inst.top_k is not None:
            cmd.extend(["--top-k", str(inst.top_k)])
        if inst.no_metrics:
            cmd.append("--no-metrics")
        if inst.spec_type:
            cmd.extend(["--spec-type", inst.spec_type])
            # Auto-detect MTP draft model for draft-mtp spec type
            if inst.spec_type == "draft-mtp":
                model_dir = Path(inst.model_path).parent
                mtp_files = sorted(model_dir.glob("mtp-*.gguf"))
                if mtp_files:
                    cmd.extend(["--model-draft", str(mtp_files[0])])
        # Auto-detect multimodal projector
        model_dir = Path(inst.model_path).parent
        mmproj_files = sorted(model_dir.glob("mmproj-*.gguf"))
        if mmproj_files:
            cmd.extend(["--mmproj", str(mmproj_files[0])])
        if inst.spec_draft_n_max is not None:
            cmd.extend(["--spec-draft-n-max", str(inst.spec_draft_n_max)])
        cmd.extend(inst.additional_args)
        return cmd

    def _kill_process(self, instance_id: str) -> None:
        proc = self._processes.pop(instance_id, None)
        if proc and proc.poll() is None:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    proc.wait(timeout=3)
            except (ProcessLookupError, OSError):
                try:
                    proc.kill()
                except (ProcessLookupError, OSError):
                    pass

    def _attach_log_stream(self, instance_id: str, pid: int) -> None:
        """Tail the log file in a background thread and keep a ring buffer."""
        if instance_id in self._log_threads:
            return  # already attached

        buf = deque(maxlen=2000)
        self._log_buffers[instance_id] = buf

        def _reader():
            log_file = self._log_dir / f"{instance_id}.log"
            while True:
                try:
                    if not log_file.exists():
                        time.sleep(0.5)
                        continue
                    with open(log_file, "r") as f:
                        # Seek to end
                        f.seek(0, 2)
                        while True:
                            line = f.readline()
                            if line:
                                buf.append(line.rstrip())
                            else:
                                # Check if process is still alive
                                if not psutil.pid_exists(pid):
                                    # Read any remaining lines
                                    while True:
                                        line = f.readline()
                                        if line:
                                            buf.append(line.rstrip())
                                        else:
                                            break
                                    return
                                time.sleep(0.3)
                except Exception:
                    return

        t = threading.Thread(target=_reader, daemon=True, name=f"log-{instance_id}")
        t.start()
        self._log_threads[instance_id] = t

    def shutdown(self) -> None:
        """Graceful shutdown — stop all processes."""
        for inst_id in list(self.instances.keys()):
            try:
                self.stop_instance(inst_id)
            except Exception:
                pass


def _human_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _title_tag(s: str) -> str:
    """Convert a kebab-case / snake_case folder segment to Title Case.

    A leading 'v' followed by digits is preserved as-is (e.g. 'v7' → 'V7').
    """
    if not s:
        return s
    # Split on hyphens and underscores
    parts = s.replace("_", "-").split("-")
    result = []
    for i, p in enumerate(parts):
        if i == 0 and p.startswith("v") and p[1:].isdigit():
            # Version prefix: "v7" → "V7", "v0.3" handled via normal title
            result.append(p.capitalize())
        else:
            result.append(p.capitalize())
    return " ".join(result)
