"""Measure real prefill and decode service times on the current device."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from microserve.generate import generate_cached
from microserve.model import Transformer


@dataclass(frozen=True)
class OperationMeasurement:
    operation: str
    shape: str
    tokens: int
    median_ms: float
    p95_ms: float
    samples: int

    @property
    def tokens_per_second(self) -> float:
        return 1_000 * self.tokens / self.median_ms


@dataclass(frozen=True)
class ServiceProfile:
    model_id: str
    device: str
    dtype: str
    prefill: tuple[OperationMeasurement, ...]
    decode: OperationMeasurement
    generation: OperationMeasurement
    prefill_tokens_per_ms: float
    prefill_fixed_ms: float
    decode_tokens_per_ms: float
    kv_bytes_per_token: int


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _measurement(
    operation: str,
    shape: str,
    tokens: int,
    durations_ms: list[float],
) -> OperationMeasurement:
    return OperationMeasurement(
        operation,
        shape,
        tokens,
        statistics.median(durations_ms),
        _p95(durations_ms),
        len(durations_ms),
    )


@torch.inference_mode()
def _measure_prefill(
    model: Transformer,
    prompt: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> OperationMeasurement:
    device = prompt.device
    durations = []
    for iteration in range(warmup + repeats):
        cache = model.new_cache(batch_size=1, max_tokens=prompt.size(1) + 1)
        _synchronize(device)
        started = time.perf_counter()
        model(prompt, cache=cache, last_token_only=True)
        _synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        cache.close()
        if iteration >= warmup:
            durations.append(elapsed_ms)
    return _measurement(
        "prefill",
        f"prompt={prompt.size(1)}",
        prompt.size(1),
        durations,
    )


@torch.inference_mode()
def _measure_decode(
    model: Transformer,
    prompt: torch.Tensor,
    *,
    decode_steps: int,
    warmup: int,
    repeats: int,
) -> OperationMeasurement:
    device = prompt.device
    durations = []
    for iteration in range(warmup + repeats):
        cache = model.new_cache(
            batch_size=1,
            max_tokens=prompt.size(1) + decode_steps,
        )
        logits = model(prompt, cache=cache, last_token_only=True)
        _synchronize(device)
        for _ in range(decode_steps):
            token = logits[:, -1].argmax(dim=-1, keepdim=True)
            _synchronize(device)
            started = time.perf_counter()
            logits = model(token, cache=cache, last_token_only=True)
            _synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1_000
            if iteration >= warmup:
                durations.append(elapsed_ms)
        cache.close()
    return _measurement(
        "decode",
        f"batch=1 context={prompt.size(1)}",
        1,
        durations,
    )


@torch.inference_mode()
def _measure_generation(
    model: Transformer,
    prompt: torch.Tensor,
    *,
    decode_steps: int,
    warmup: int,
    repeats: int,
) -> OperationMeasurement:
    """Measure real end-to-end cached generation, including prefill."""
    device = prompt.device
    durations = []
    for iteration in range(warmup + repeats):
        _synchronize(device)
        started = time.perf_counter()
        generate_cached(model, prompt, max_new_tokens=decode_steps)
        _synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        if iteration >= warmup:
            durations.append(elapsed_ms)
    return _measurement(
        "generate",
        f"prompt={prompt.size(1)} output={decode_steps}",
        decode_steps,
        durations,
    )


def _fit_prefill(
    measurements: Sequence[OperationMeasurement],
) -> tuple[float, float]:
    """Fit ``milliseconds = fixed + tokens / rate`` with nonnegative terms."""
    if len(measurements) == 1:
        point = measurements[0]
        return point.tokens / point.median_ms, 0.0

    xs = [float(point.tokens) for point in measurements]
    ys = [point.median_ms for point in measurements]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    unconstrained_slope = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)
    ) / denominator
    unconstrained_fixed = mean_y - unconstrained_slope * mean_x

    # With two nonnegative coefficients, the least-squares optimum is either
    # the unconstrained fit or one of the axes. Tiny prefills are often dominated
    # by launch overhead, and measurement noise can otherwise imply a nonsensical
    # negative per-token cost.
    candidates = [
        (0.0, mean_y),
        (sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs), 0.0),
    ]
    if unconstrained_slope >= 0 and unconstrained_fixed >= 0:
        candidates.append((unconstrained_slope, unconstrained_fixed))
    slope, fixed = min(
        candidates,
        key=lambda pair: sum(
            (y - (pair[1] + pair[0] * x)) ** 2 for x, y in zip(xs, ys)
        ),
    )
    rate = float("inf") if slope == 0 else 1 / slope
    return rate, fixed


def kv_bytes_per_token(model: Transformer) -> int:
    config = model.config
    element_size = next(model.parameters()).element_size()
    return (
        2
        * config.num_layers
        * config.num_kv_heads
        * config.head_dim
        * element_size
    )


def profile_model(
    model: Transformer,
    *,
    model_id: str,
    prompt_lengths: Sequence[int] = (8, 32, 56),
    decode_steps: int = 8,
    warmup: int = 1,
    repeats: int = 3,
) -> ServiceProfile:
    """Run synchronized model calls and derive a simulator service profile."""
    if not prompt_lengths or min(prompt_lengths) < 1:
        raise ValueError("prompt lengths must be positive")
    if decode_steps < 1 or warmup < 0 or repeats < 1:
        raise ValueError("invalid profiling iteration counts")
    if max(prompt_lengths) + decode_steps > model.config.max_seq_len:
        raise ValueError("profile workload exceeds the model context limit")

    parameter = next(model.parameters())
    device = parameter.device
    generator = torch.Generator()
    generator.manual_seed(0)
    prompts = {
        length: torch.randint(
            model.config.vocab_size,
            (1, length),
            generator=generator,
        ).to(device)
        for length in sorted(set(prompt_lengths))
    }
    prefill = tuple(
        _measure_prefill(model, prompt, warmup=warmup, repeats=repeats)
        for prompt in prompts.values()
    )
    decode = _measure_decode(
        model,
        prompts[max(prompts)],
        decode_steps=decode_steps,
        warmup=warmup,
        repeats=repeats,
    )
    generation = _measure_generation(
        model,
        prompts[max(prompts)],
        decode_steps=decode_steps,
        warmup=warmup,
        repeats=repeats,
    )
    prefill_rate, prefill_fixed = _fit_prefill(prefill)
    return ServiceProfile(
        model_id=model_id,
        device=str(device),
        dtype=str(parameter.dtype),
        prefill=prefill,
        decode=decode,
        generation=generation,
        prefill_tokens_per_ms=prefill_rate,
        prefill_fixed_ms=prefill_fixed,
        decode_tokens_per_ms=1 / decode.median_ms,
        kv_bytes_per_token=kv_bytes_per_token(model),
    )


def synthetic_service_profile() -> ServiceProfile:
    """Small deterministic profile used by API tests, never by the CLI."""
    prefill = (
        OperationMeasurement("prefill", "prompt=8", 8, 2.0, 2.0, 1),
        OperationMeasurement("prefill", "prompt=32", 32, 8.0, 8.0, 1),
        OperationMeasurement("prefill", "prompt=56", 56, 14.0, 14.0, 1),
    )
    decode = OperationMeasurement("decode", "batch=1 context=56", 1, 1.0, 1.0, 1)
    generation = OperationMeasurement(
        "generate", "prompt=56 output=8", 8, 22.0, 22.0, 1
    )
    return ServiceProfile(
        model_id="synthetic service profile",
        device="simulator",
        dtype="n/a",
        prefill=prefill,
        decode=decode,
        generation=generation,
        prefill_tokens_per_ms=4.0,
        prefill_fixed_ms=0.0,
        decode_tokens_per_ms=1.0,
        kv_bytes_per_token=512,
    )
