import torch

from microserve import ModelConfig, Transformer
from microserve.profiler import (
    OperationMeasurement,
    _fit_prefill,
    kv_bytes_per_token,
    profile_model,
)


def test_profile_model_measures_actual_prefill_and_decode_calls() -> None:
    torch.manual_seed(53)
    model = Transformer(
        ModelConfig(
            vocab_size=16,
            dim=8,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            hidden_dim=16,
            max_seq_len=16,
        )
    ).eval()

    profile = profile_model(
        model,
        model_id="test-model",
        prompt_lengths=(2, 4),
        decode_steps=2,
        warmup=0,
        repeats=1,
    )

    assert profile.model_id == "test-model"
    assert [measurement.tokens for measurement in profile.prefill] == [2, 4]
    assert all(measurement.median_ms > 0 for measurement in profile.prefill)
    assert profile.decode.samples == 2
    assert profile.decode.median_ms > 0
    assert profile.generation.samples == 1
    assert profile.generation.tokens == 2
    assert profile.generation.median_ms > 0
    assert profile.prefill_tokens_per_ms > 0
    assert profile.decode_tokens_per_ms > 0
    assert profile.kv_bytes_per_token == kv_bytes_per_token(model) == 32


def test_prefill_fit_treats_overhead_dominated_noise_as_constant() -> None:
    measurements = [
        OperationMeasurement("prefill", "prompt=8", 8, 20.0, 20.0, 3),
        OperationMeasurement("prefill", "prompt=32", 32, 21.0, 21.0, 3),
        OperationMeasurement("prefill", "prompt=56", 56, 19.0, 19.0, 3),
    ]

    rate, fixed = _fit_prefill(measurements)

    assert rate == float("inf")
    assert fixed == 20.0
