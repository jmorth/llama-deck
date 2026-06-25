"""GPU detection and monitoring via nvidia-smi."""

from __future__ import annotations

import subprocess
from typing import List

from .models import GPUInfo, GPUStatus


def _parse_nvidia_smi() -> GPUStatus:
    """Query nvidia-smi and parse the output into GPUInfo objects."""
    try:
        # Query format: index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return GPUStatus(
                gpus=[],
                available=False,
                error=f"nvidia-smi failed: {result.stderr.strip()}",
            )

        gpus: List[GPUInfo] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue

            def _float(val: str) -> float:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0

            def _int(val: str) -> int | None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return None

            gpu = GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                memory_total_mb=_float(parts[2]),
                memory_used_mb=_float(parts[3]),
                memory_free_mb=_float(parts[4]),
                utilization_percent=_float(parts[5]),
                temperature=_int(parts[6]) if len(parts) > 6 else None,
                power_draw_w=_float(parts[7]) if len(parts) > 7 else None,
            )
            gpus.append(gpu)

        # Get driver and CUDA versions
        driver_version = ""
        cuda_version = ""
        try:
            ver_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if ver_result.returncode == 0 and ver_result.stdout.strip():
                driver_version = ver_result.stdout.strip().splitlines()[0]
        except Exception:
            pass

        try:
            ver_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if ver_result.returncode == 0 and ver_result.stdout.strip():
                cuda_version = ver_result.stdout.strip().splitlines()[0]
        except Exception:
            pass

        return GPUStatus(
            gpus=gpus,
            driver_version=driver_version,
            cuda_version=cuda_version,
            available=True,
        )

    except FileNotFoundError:
        return GPUStatus(
            gpus=[],
            available=False,
            error="nvidia-smi not found — no NVIDIA drivers installed",
        )
    except subprocess.TimeoutExpired:
        return GPUStatus(
            gpus=[],
            available=False,
            error="nvidia-smi timed out",
        )
    except Exception as e:
        return GPUStatus(
            gpus=[],
            available=False,
            error=str(e),
        )


def get_gpu_status() -> GPUStatus:
    """Get current GPU status. Returns empty list with error if nvidia-smi unavailable."""
    return _parse_nvidia_smi()


def get_gpu_indices(gpu_status: GPUStatus) -> List[int]:
    """Return list of GPU indices available on the system."""
    return [gpu.index for gpu in gpu_status.gpus]
