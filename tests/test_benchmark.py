import pytest
import torch

from microserve import ModelConfig, Transformer, generate_naive
from microserve.benchmark import measure


def test_measure_reports_generated_token_rate() -> None:
    torch.manual_seed(3)
    model = Transformer(
        ModelConfig(
            vocab_size=16,
            dim=16,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            max_seq_len=8,
        )
    ).eval()
    prompt = torch.tensor([[1, 2], [3, 4]])

    result = measure(
        generate_naive,
        model,
        prompt,
        max_new_tokens=2,
        warmup=0,
        repeats=1,
    )

    assert result.median_seconds > 0
    assert result.tokens_per_second == pytest.approx(4 / result.median_seconds)
