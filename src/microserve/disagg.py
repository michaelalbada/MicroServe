"""Stage 7: a deterministic prefill/decode disaggregation simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass

from microserve.router import ReplicaState, RouteRequest, SLOAwareRouter


@dataclass(frozen=True)
class WorkloadRequest:
    request_id: str
    prompt_tokens: int
    output_tokens: int
    arrival_ms: float = 0.0
    ttft_slo_ms: float | None = None
    itl_slo_ms: float | None = None

    def __post_init__(self) -> None:
        if self.prompt_tokens < 1 or self.output_tokens < 1:
            raise ValueError("prompt and output token counts must be positive")
        if self.arrival_ms < 0:
            raise ValueError("arrival time must be non-negative")


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    tokens_per_ms: float
    fixed_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.tokens_per_ms <= 0 or self.fixed_ms < 0:
            raise ValueError("worker rate must be positive and fixed cost non-negative")


@dataclass
class _Worker:
    spec: WorkerSpec
    available_at_ms: float = 0.0

    def snapshot(self) -> ReplicaState:
        return ReplicaState(
            self.spec.name,
            self.available_at_ms,
            self.spec.tokens_per_ms,
            self.spec.fixed_ms,
        )


@dataclass(frozen=True)
class DisaggregatedResult:
    request_id: str
    arrival_ms: float
    prefill_queue_ms: float
    prefill_compute_ms: float
    transfer_queue_ms: float
    kv_transfer_ms: float
    decode_queue_ms: float
    token_times_ms: tuple[float, ...]
    ttft_slo_ms: float | None
    itl_slo_ms: float | None

    @property
    def ttft_ms(self) -> float:
        return self.token_times_ms[0] - self.arrival_ms

    @property
    def e2e_ms(self) -> float:
        return self.token_times_ms[-1] - self.arrival_ms

    @property
    def itl_ms(self) -> tuple[float, ...]:
        return tuple(
            right - left
            for left, right in zip(self.token_times_ms, self.token_times_ms[1:])
        )

    @property
    def meets_slo(self) -> bool:
        ttft_ok = self.ttft_slo_ms is None or self.ttft_ms <= self.ttft_slo_ms
        itl_ok = self.itl_slo_ms is None or all(
            latency <= self.itl_slo_ms for latency in self.itl_ms
        )
        return ttft_ok and itl_ok


@dataclass(frozen=True)
class LatencySummary:
    requests: int
    p50_ttft_ms: float
    p95_ttft_ms: float
    p95_itl_ms: float
    slo_attainment: float
    makespan_ms: float
    output_tokens_per_second: float


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize(
    requests: list[WorkloadRequest], results: dict[str, DisaggregatedResult]
) -> LatencySummary:
    if not requests:
        return LatencySummary(0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    ttfts = [results[request.request_id].ttft_ms for request in requests]
    itls = [
        latency
        for request in requests
        for latency in results[request.request_id].itl_ms
    ]
    start = min(request.arrival_ms for request in requests)
    end = max(results[request.request_id].token_times_ms[-1] for request in requests)
    makespan = end - start
    output_tokens = sum(request.output_tokens for request in requests)
    return LatencySummary(
        requests=len(requests),
        p50_ttft_ms=_percentile(ttfts, 0.50),
        p95_ttft_ms=_percentile(ttfts, 0.95),
        p95_itl_ms=_percentile(itls, 0.95),
        slo_attainment=sum(result.meets_slo for result in results.values())
        / len(results),
        makespan_ms=makespan,
        output_tokens_per_second=1_000 * output_tokens / makespan,
    )


class DisaggregatedSimulator:
    """Model separate prefill, KV-transfer, and decode queues.

    It is intentionally a cost simulator rather than a fake distributed RPC
    layer. That makes queueing and transport assumptions inspectable and keeps
    tests deterministic.
    """

    def __init__(
        self,
        *,
        prefill_workers: list[WorkerSpec],
        decode_workers: list[WorkerSpec],
        kv_bytes_per_token: int,
        transfer_bandwidth_bytes_per_ms: float,
        transfer_fixed_ms: float = 0.0,
        slo_aware: bool = False,
    ) -> None:
        if not prefill_workers or not decode_workers:
            raise ValueError("prefill and decode each need at least one worker")
        if kv_bytes_per_token < 0 or transfer_bandwidth_bytes_per_ms <= 0:
            raise ValueError("invalid KV transfer parameters")
        self.prefill_specs = prefill_workers
        self.decode_specs = decode_workers
        self.kv_bytes_per_token = kv_bytes_per_token
        self.transfer_bandwidth = transfer_bandwidth_bytes_per_ms
        self.transfer_fixed_ms = transfer_fixed_ms
        self.slo_aware = slo_aware
        self.router = SLOAwareRouter()

    def _order(self, requests: list[WorkloadRequest]) -> list[WorkloadRequest]:
        if not self.slo_aware:
            return sorted(requests, key=lambda item: (item.arrival_ms, item.request_id))
        return sorted(
            requests,
            key=lambda item: (
                item.arrival_ms,
                item.arrival_ms + item.ttft_slo_ms
                if item.ttft_slo_ms is not None
                else float("inf"),
                item.request_id,
            ),
        )

    def run(self, requests: list[WorkloadRequest]) -> dict[str, DisaggregatedResult]:
        prefill = [_Worker(spec) for spec in self.prefill_specs]
        decode = [_Worker(spec) for spec in self.decode_specs]
        transfer_available = 0.0
        results = {}

        for request in self._order(requests):
            deadline = (
                request.arrival_ms + request.ttft_slo_ms
                if request.ttft_slo_ms is not None
                else None
            )
            prefill_decision = self.router.route(
                RouteRequest(request.arrival_ms, request.prompt_tokens, deadline),
                [worker.snapshot() for worker in prefill],
            )
            prefill_worker = next(
                worker
                for worker in prefill
                if worker.spec.name == prefill_decision.replica.name
            )
            prefill_start = prefill_decision.predicted_start_ms
            prefill_compute = (
                prefill_worker.spec.fixed_ms
                + request.prompt_tokens / prefill_worker.spec.tokens_per_ms
            )
            prefill_end = prefill_start + prefill_compute
            prefill_worker.available_at_ms = prefill_end

            transfer_start = max(prefill_end, transfer_available)
            transfer_time = (
                self.transfer_fixed_ms
                + (request.prompt_tokens * self.kv_bytes_per_token)
                / self.transfer_bandwidth
            )
            transfer_end = transfer_start + transfer_time
            transfer_available = transfer_end

            # A decode worker becomes eligible only after its KV state arrives.
            # Its route prediction uses one token because that is the TTFT event.
            decode_decision = self.router.route(
                RouteRequest(transfer_end, 1, deadline),
                [worker.snapshot() for worker in decode],
            )
            decode_worker = next(
                worker
                for worker in decode
                if worker.spec.name == decode_decision.replica.name
            )
            decode_start = decode_decision.predicted_start_ms
            per_token = 1 / decode_worker.spec.tokens_per_ms
            first_token = decode_start + decode_worker.spec.fixed_ms + per_token
            token_times = tuple(
                first_token + index * per_token
                for index in range(request.output_tokens)
            )
            decode_worker.available_at_ms = token_times[-1]

            results[request.request_id] = DisaggregatedResult(
                request_id=request.request_id,
                arrival_ms=request.arrival_ms,
                prefill_queue_ms=prefill_start - request.arrival_ms,
                prefill_compute_ms=prefill_compute,
                transfer_queue_ms=transfer_start - prefill_end,
                kv_transfer_ms=transfer_time,
                decode_queue_ms=decode_start - transfer_end,
                token_times_ms=token_times,
                ttft_slo_ms=request.ttft_slo_ms,
                itl_slo_ms=request.itl_slo_ms,
            )
        return results
