import pytest

from microserve import (
    DisaggregatedSimulator,
    QueueOnlyRouter,
    ReplicaState,
    RouteRequest,
    SLOAwareRouter,
    WorkerSpec,
    WorkloadRequest,
    summarize,
)


def test_disaggregated_latency_breakdown_is_explicit() -> None:
    simulator = DisaggregatedSimulator(
        prefill_workers=[WorkerSpec("p0", tokens_per_ms=10)],
        decode_workers=[WorkerSpec("d0", tokens_per_ms=2)],
        kv_bytes_per_token=100,
        transfer_bandwidth_bytes_per_ms=1_000,
        transfer_fixed_ms=1,
    )
    request = WorkloadRequest("r", prompt_tokens=20, output_tokens=3)

    result = simulator.run([request])["r"]

    assert result.prefill_queue_ms == 0
    assert result.prefill_compute_ms == 2
    assert result.kv_transfer_ms == 3
    assert result.decode_queue_ms == 0
    assert result.ttft_ms == pytest.approx(5.5)
    assert result.itl_ms == pytest.approx((0.5, 0.5))
    assert result.e2e_ms == pytest.approx(6.5)


def test_single_transfer_link_exposes_queueing() -> None:
    simulator = DisaggregatedSimulator(
        prefill_workers=[
            WorkerSpec("p0", tokens_per_ms=100),
            WorkerSpec("p1", tokens_per_ms=100),
        ],
        decode_workers=[WorkerSpec("d0", tokens_per_ms=10)],
        kv_bytes_per_token=1_000,
        transfer_bandwidth_bytes_per_ms=1_000,
    )
    requests = [
        WorkloadRequest("a", prompt_tokens=10, output_tokens=1),
        WorkloadRequest("b", prompt_tokens=10, output_tokens=1),
    ]

    results = simulator.run(requests)

    assert results["b"].transfer_queue_ms == pytest.approx(10)
    assert results["b"].kv_transfer_ms > results["b"].prefill_compute_ms


def test_slo_aware_order_protects_urgent_ttft() -> None:
    requests = [
        WorkloadRequest("relaxed", 100, 1, ttft_slo_ms=100),
        WorkloadRequest("urgent", 1, 1, ttft_slo_ms=5),
    ]
    arguments = dict(
        prefill_workers=[WorkerSpec("p0", tokens_per_ms=1)],
        decode_workers=[WorkerSpec("d0", tokens_per_ms=10)],
        kv_bytes_per_token=0,
        transfer_bandwidth_bytes_per_ms=1,
    )

    fcfs = DisaggregatedSimulator(**arguments).run(requests)
    aware = DisaggregatedSimulator(**arguments, slo_aware=True).run(requests)

    assert fcfs["urgent"].ttft_ms > 100
    assert aware["urgent"].ttft_ms < 5
    assert aware["urgent"].meets_slo


def test_predictive_router_values_prefix_locality_over_empty_queue() -> None:
    request = RouteRequest(ready_at_ms=0, tokens=100, deadline_ms=20)
    empty = ReplicaState("empty", 0, tokens_per_ms=5)
    cached = ReplicaState("cached", 5, tokens_per_ms=5, cached_prefix_tokens=90)

    assert QueueOnlyRouter().route(request, [empty, cached]).replica is empty
    decision = SLOAwareRouter().route(request, [empty, cached])
    assert decision.replica is cached
    assert decision.predicted_deadline_miss_ms == 0


def test_summary_reports_tail_latency_throughput_and_slo() -> None:
    simulator = DisaggregatedSimulator(
        prefill_workers=[WorkerSpec("p0", tokens_per_ms=10)],
        decode_workers=[WorkerSpec("d0", tokens_per_ms=2)],
        kv_bytes_per_token=0,
        transfer_bandwidth_bytes_per_ms=1,
    )
    requests = [
        WorkloadRequest("a", 10, 2, ttft_slo_ms=2, itl_slo_ms=1),
        WorkloadRequest("b", 20, 2, ttft_slo_ms=2, itl_slo_ms=1),
    ]
    results = simulator.run(requests)

    summary = summarize(requests, results)

    assert summary.requests == 2
    assert summary.p95_ttft_ms >= summary.p50_ttft_ms
    assert summary.p95_itl_ms == pytest.approx(0.5)
    assert 0 <= summary.slo_attainment <= 1
    assert summary.output_tokens_per_second > 0
