"""A small scorecard for comparing serving stages on one workload."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from microserve.generate import generate_cached, generate_naive
from microserve.model import ModelConfig, Transformer


Generate = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class Measurement:
    median_seconds: float
    tokens_per_second: float


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure(
    generate: Generate,
    model: Transformer,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    warmup: int,
    repeats: int,
) -> Measurement:
    """Measure steady-state end-to-end generation for a fixed batch."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    for _ in range(warmup):
        generate(model, prompt, max_new_tokens=max_new_tokens)
    _synchronize(prompt.device)

    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        generate(model, prompt, max_new_tokens=max_new_tokens)
        _synchronize(prompt.device)
        durations.append(time.perf_counter() - started)

    median_seconds = statistics.median(durations)
    generated_tokens = prompt.size(0) * max_new_tokens
    return Measurement(
        median_seconds=median_seconds,
        tokens_per_second=generated_tokens / median_seconds,
    )


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare naive and KV-cached autoregressive generation."
    )
    parser.add_argument("--device", default=str(_default_device()))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=256,
        dim=256,
        num_layers=6,
        num_heads=8,
        num_kv_heads=2,
        max_seq_len=args.prompt_tokens + args.new_tokens,
    )
    model = Transformer(config).eval().to(device)
    prompt = torch.randint(
        config.vocab_size,
        (args.batch_size, args.prompt_tokens),
        device=device,
    )

    methods = {
        "naive": generate_naive,
        "kv_cache": generate_cached,
    }
    results = {
        name: measure(
            generate,
            model,
            prompt,
            max_new_tokens=args.new_tokens,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for name, generate in methods.items()
    }

    reference = methods["naive"](model, prompt, max_new_tokens=args.new_tokens)
    cached = methods["kv_cache"](model, prompt, max_new_tokens=args.new_tokens)
    torch.testing.assert_close(cached, reference)

    print(
        f"device={device} batch={args.batch_size} "
        f"prompt={args.prompt_tokens} decode={args.new_tokens}"
    )
    print("stage       median (ms)    generated tok/s    speedup")
    baseline = results["naive"].median_seconds
    for name, result in results.items():
        print(
            f"{name:<11} {result.median_seconds * 1_000:>11.2f} "
            f"{result.tokens_per_second:>18.2f} "
            f"{baseline / result.median_seconds:>9.2f}x"
        )


if __name__ == "__main__":
    main()
