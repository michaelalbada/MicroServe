import pytest
import torch

from microserve.memory import (
    MemoryBudgetError,
    estimate_memory,
    parameter_count,
    require_memory_budget,
)
from microserve.model import ModelConfig, Transformer


def benchmark_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        dim=256,
        num_layers=6,
        num_heads=8,
        num_kv_heads=2,
        max_seq_len=2_000,
    )


def test_naive_estimate_explains_observed_29_gib_attention_buffer() -> None:
    estimate = estimate_memory(
        benchmark_config(),
        method="naive",
        batch_size=1_000,
        prompt_tokens=1_000,
        new_tokens=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        available_bytes=128 * 1024**3,
    )

    assert estimate.attention_bytes == 1_000 * 8 * 1_000**2 * 4
    assert estimate.attention_bytes / 1024**3 == pytest.approx(29.8, rel=0.01)


def test_kv_estimate_uses_layers_heads_and_full_capacity() -> None:
    config = benchmark_config()
    estimate = estimate_memory(
        config,
        method="kv_cache",
        batch_size=10,
        prompt_tokens=100,
        new_tokens=100,
        dtype=torch.float16,
        device=torch.device("cpu"),
        available_bytes=64 * 1024**3,
    )

    expected = 10 * 200 * 6 * 2 * 2 * 32 * 2
    assert estimate.kv_cache_bytes == expected
    assert estimate.fits


def test_memory_budget_refuses_unsafe_run_unless_forced() -> None:
    estimate = estimate_memory(
        benchmark_config(),
        method="naive",
        batch_size=1_000,
        prompt_tokens=1_000,
        new_tokens=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        available_bytes=8 * 1024**3,
    )

    with pytest.raises(MemoryBudgetError):
        require_memory_budget([estimate])
    require_memory_budget([estimate], force=True)


def test_parameter_formula_matches_the_actual_tied_model() -> None:
    config = ModelConfig(
        vocab_size=16,
        dim=16,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        hidden_dim=32,
    )
    model = Transformer(config)

    assert parameter_count(config) == sum(
        parameter.numel() for parameter in model.parameters()
    )
