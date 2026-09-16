from microserve.scorecard import run_curriculum_scorecard, run_system_scorecard
from microserve.profiler import synthetic_service_profile


def scorecard():
    return run_system_scorecard(synthetic_service_profile())


def test_scorecard_compares_system_configurations_on_one_workload() -> None:
    result = scorecard()

    assert len(result.workload) == 12
    assert len(result.configurations) == 8
    assert result.execution == ()
    assert result.system == result.configurations
    assert all(item.summary.requests == 12 for item in result.configurations)


def test_configuration_sweep_exposes_policy_and_capacity_tradeoffs() -> None:
    results = {
        result.configuration.name: result for result in scorecard().system
    }

    assert (
        results["slo_policy"].summary.slo_attainment
        > results["baseline"].summary.slo_attainment
    )
    assert (
        results["decode_heavy"].summary.output_tokens_per_second
        > results["prefill_heavy"].summary.output_tokens_per_second
    )
    assert (
        results["balanced"].p95_transfer_ms
        < results["balanced_slow_link"].p95_transfer_ms
    )
    assert (
        results["scaled_fast_link"].summary.slo_attainment
        > results["balanced"].summary.slo_attainment
    )


def test_old_scorecard_entry_point_remains_available() -> None:
    assert run_curriculum_scorecard(device="mps") == scorecard()
