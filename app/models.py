"""Data models for LlamaDeck."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class InstanceStatus(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class GPUInfo(BaseModel):
    index: int
    name: str
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    utilization_percent: float
    temperature: Optional[int] = None
    power_draw_w: Optional[float] = None


class GPUStatus(BaseModel):
    gpus: List[GPUInfo]
    driver_version: str = ""
    cuda_version: str = ""
    available: bool = True
    error: Optional[str] = None


class LlamaInstance(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    model_path: str
    gpus: List[int] = Field(default_factory=list)
    gpu_count: Optional[int] = None  # mutually exclusive with gpus — auto-assign N GPUs on start
    port: int
    host: str = "0.0.0.0"
    status: InstanceStatus = InstanceStatus.STOPPED
    pid: Optional[int] = None
    context_size: int = 4096
    n_gpu_layers: int = 999
    parallel: int = 4
    flash_attention: bool = True
    additional_args: List[str] = Field(default_factory=list)
    created_at: str = ""
    started_at: Optional[str] = None
    error_message: Optional[str] = None

    @model_validator(mode="after")
    def _check_gpu_exclusivity(self) -> "LlamaInstance":
        if self.gpus and self.gpu_count is not None:
            raise ValueError("gpus (pinned) and gpu_count (auto-assign) are mutually exclusive")
        return self


class InstanceCreate(BaseModel):
    name: str
    model_path: str
    gpus: List[int] = Field(default_factory=list)
    gpu_count: Optional[int] = None  # mutually exclusive with gpus
    port: Optional[int] = None  # auto-assign if None
    host: str = "0.0.0.0"
    context_size: int = 4096
    n_gpu_layers: int = 999
    parallel: int = 4
    flash_attention: bool = True
    additional_args: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_gpu_exclusivity(self) -> "InstanceCreate":
        if self.gpus and self.gpu_count is not None:
            raise ValueError("gpus (pinned) and gpu_count (auto-assign) are mutually exclusive")
        return self


class InstanceUpdate(BaseModel):
    name: Optional[str] = None
    model_path: Optional[str] = None
    gpus: Optional[List[int]] = None
    gpu_count: Optional[int] = None
    port: Optional[int] = None
    host: Optional[str] = None
    context_size: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    parallel: Optional[int] = None
    flash_attention: Optional[bool] = None
    additional_args: Optional[List[str]] = None

    @model_validator(mode="after")
    def _check_gpu_exclusivity(self) -> "InstanceUpdate":
        if self.gpus is not None and self.gpu_count is not None:
            raise ValueError("gpus (pinned) and gpu_count (auto-assign) are mutually exclusive")
        return self


class Config(BaseModel):
    models_dir: str = "/models"
    llama_server_binary: str = "llama-server"
    default_host: str = "0.0.0.0"
    port_range_start: int = 8081
    port_range_end: int = 9000
    data_dir: str = "data"


class ModelFile(BaseModel):
    name: str
    path: str
    size_bytes: int
    size_human: str


class OverallStatus(BaseModel):
    gpu_status: GPUStatus
    instances: List[LlamaInstance]
    model_files: List[ModelFile]
    config: Config
