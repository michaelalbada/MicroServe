"""Memory estimates and device cleanup for benchmark preflight."""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass

import psutil
import torch

from microserve.model import ModelConfig, Transformer


@dataclass(frozen=True)
class MemoryEstimate:
    method: str
    model_bytes: int
    kv_cache_bytes: int
    attention_bytes: int
    activation_bytes: int
    allocator_churn_bytes: int
    estimated_peak_bytes: int
    available_bytes: int
    memory_fraction: float

    @property
    def limit_bytes(self) -> int:
        return int(self.available_bytes * self.memory_fraction)

    @property
    def fits(self) -> bool:
        return self.estimated_peak_bytes <= self.limit_bytes


class MemoryBudgetError(RuntimeError):
    def __init__(self, estimates: list[MemoryEstimate]) -> None:
        self.estimates = estimates
        failed = ", ".join(item.method for item in estimates if not item.fits)
        super().__init__(f"estimated memory exceeds the safe budget for: {failed}")


def dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def parameter_count(config: ModelConfig) -> int:
    attention = (
        2 * config.dim * config.dim
        + 2 * config.dim * config.num_kv_heads * config.head_dim
    )
    feed_forward = 3 * config.dim * config.mlp_dim
    norms = 2 * config.dim
    return (
        config.vocab_size * config.dim
        + config.num_layers * (attention + feed_forward + norms)
        + config.dim
    )


def device_available_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return int(free)
    # MPS shares system memory. Avoid allocator introspection here: some Metal
    # driver states can crash inside recommended_max_memory() after a prior OOM,
    # while psutil remains safe and is sufficient for a conservative preflight.
    if device.type == "mps":
        return int(psutil.virtual_memory().available)
    return int(psutil.virtual_memory().available)


def estimate_memory(
    config: ModelConfig,
    *,
    method: str,
    batch_size: int,
    prompt_tokens: int,
    new_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    available_bytes: int | None = None,
    memory_fraction: float = 0.75,
) -> MemoryEstimate:
    if method not in {"naive", "kv_cache"}:
        raise ValueError(f"unknown generation method: {method}")
    if min(batch_size, prompt_tokens) < 1 or new_tokens < 0:
        raise ValueError("batch/prompt must be positive and new_tokens non-negative")
    if not 0 < memory_fraction <= 1:
        raise ValueError("memory_fraction must be in (0, 1]")

    element = dtype_bytes(dtype)
    max_context = prompt_tokens + max(0, new_tokens - 1)
    model_bytes = parameter_count(config) * element
    kv_cache_bytes = 0
    if method == "kv_cache":
        kv_cache_bytes = (
            batch_size
            * (prompt_tokens + new_tokens)
            * config.num_layers
            * 2
            * config.num_kv_heads
            * config.head_dim
            * element
        )

    attention_context = max_context if method == "naive" else prompt_tokens
    attention_bytes = (
        batch_size * config.num_heads * attention_context * attention_context * element
    )
    activation_bytes = (
        batch_size
        * attention_context
        * (config.num_heads + 2 * config.num_kv_heads)
        * config.head_dim
        * element
    )
    # MPS caches a series of differently sized SDPA workspaces during naive
    # generation. Sum those shapes so preflight reflects the observed allocator
    # pressure instead of reporting only the final step's theoretical peak.
    allocator_churn = 0
    if device.type == "mps" and method == "naive" and new_tokens:
        squared_contexts = sum(
            length * length
            for length in range(prompt_tokens, prompt_tokens + new_tokens)
        )
        allocator_churn = batch_size * config.num_heads * squared_contexts * element

    workspace = max(attention_bytes, allocator_churn)
    estimated_peak = math.ceil(
        1.20 * (model_bytes + kv_cache_bytes + activation_bytes + workspace)
    )
    return MemoryEstimate(
        method=method,
        model_bytes=model_bytes,
        kv_cache_bytes=kv_cache_bytes,
        attention_bytes=attention_bytes,
        activation_bytes=activation_bytes,
        allocator_churn_bytes=allocator_churn,
        estimated_peak_bytes=estimated_peak,
        available_bytes=available_bytes
        if available_bytes is not None
        else device_available_bytes(device),
        memory_fraction=memory_fraction,
    )


def estimate_model_memory(model: Transformer) -> int:
    return sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )


def require_memory_budget(
    estimates: list[MemoryEstimate], *, force: bool = False
) -> None:
    if not force and any(not estimate.fits for estimate in estimates):
        raise MemoryBudgetError(estimates)


def release_device_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")
