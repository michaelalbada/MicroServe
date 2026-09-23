import torch

from microserve import ModelConfig, Transformer
from microserve.scorecard import (
    EXECUTION_CONFIGURATIONS,
    _runtime_requests,
    measure_execution_configuration,
)


def test_executable_scorecard_runs_every_configuration_and_checks_outputs() -> None:
    torch.manual_seed(5)
    model = Transformer(
        ModelConfig(
            vocab_size=32,
            dim=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            max_seq_len=32,
        )
    ).eval()
    requests = _runtime_requests(
        model,
        prompt_lengths=(3, 5),
        decode_steps=3,
    )
    reference = None
    measurements = []

    for configuration in EXECUTION_CONFIGURATIONS:
        measurement, outputs = measure_execution_configuration(
            model,
            requests,
            configuration,
            reference=reference,
            warmup=0,
            repeats=1,
            block_size=2,
            chunk_size=2,
        )
        if reference is None:
            reference = outputs
        measurements.append(measurement)

    assert all(measurement.correct for measurement in measurements)
    assert all(measurement.median_wall_ms > 0 for measurement in measurements)
    assert all(measurement.output_tokens_per_second > 0 for measurement in measurements)
    assert measurements[0].peak_blocks is None
    assert all(measurement.peak_blocks for measurement in measurements[2:])
