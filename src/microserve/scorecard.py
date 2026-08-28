"""Run one coherent workload across the complete microServe curriculum."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from microserve.allocator import BlockAllocator, PagedKVCache
from microserve.batcher import ContinuousBatcher, Request
from microserve.disagg import (
    DisaggregatedSimulator,
    LatencySummary,
    WorkerSpec,
    WorkloadRequest,
    summarize,
)
from microserve.generate import generate_cached, generate_naive
from microserve.model import ModelConfig, Transformer
from microserve.prefix_cache import PrefixCache
from microserve.scheduler import FCFSScheduler
from microserve.speculative import generate_speculative


@dataclass(frozen=True)
class ExecutionStage:
    stage: int
    name: str
    correct: bool
    wall_ms: float
    output_tokens: int
    engine_steps: int | None = None
    mean_ttft_steps: float | None = None
    p95_itl_steps: float | None = None
    peak_kv_blocks: int | None = None
    prefix_hits: int | None = None
    target_calls: int | None = None

    @property
    def output_tokens_per_second(self) -> float:
        return 1_000 * self.output_tokens / self.wall_ms


@dataclass(frozen=True)
class SystemStage:
    stage: int
    name: str
    summary: LatencySummary
    p95_transfer_ms: float
    p95_queue_ms: float


@dataclass(frozen=True)
class CurriculumScorecard:
    execution: tuple[ExecutionStage, ...]
    system: tuple[SystemStage, ...]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _optional(value: int | float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _workload(device: torch.device) -> list[Request]:
    return [
        Request(
            "a_shared",
            torch.tensor([1, 2, 3, 4, 5], device=device),
            4,
            arrival_step=0,
            ttft_slo_steps=20,
            itl_slo_steps=2,
        ),
        Request(
            "b_urgent",
            torch.tensor([9, 10, 11], device=device),
            3,
            arrival_step=0,
            ttft_slo_steps=6,
            itl_slo_steps=2,
        ),
        Request(
            "c_shared",
            torch.tensor([1, 2, 3, 4, 6, 7], device=device),
            4,
            arrival_step=2,
            ttft_slo_steps=20,
            itl_slo_steps=2,
        ),
    ]


def _references(
    model: Transformer, requests: list[Request]
) -> dict[str, tuple[int, ...]]:
    return {
        request.request_id: tuple(
            generate_naive(
                model,
                request.prompt[None],
                max_new_tokens=request.max_new_tokens,
            )[0, request.prompt.numel() :].tolist()
        )
        for request in requests
    }


def _serial_stage(
    stage: int,
    name: str,
    generate,
    model: Transformer,
    requests: list[Request],
    references: dict[str, tuple[int, ...]],
) -> ExecutionStage:
    started = time.perf_counter()
    outputs = {}
    for request in requests:
        tokens = generate(
            model,
            request.prompt[None],
            max_new_tokens=request.max_new_tokens,
        )
        outputs[request.request_id] = tuple(
            tokens[0, request.prompt.numel() :].tolist()
        )
    wall_ms = (time.perf_counter() - started) * 1_000
    return ExecutionStage(
        stage,
        name,
        outputs == references,
        wall_ms,
        sum(request.max_new_tokens for request in requests),
    )


def _engine_stage(
    stage: int,
    name: str,
    batcher: ContinuousBatcher,
    requests: list[Request],
    references: dict[str, tuple[int, ...]],
    *,
    allocator: BlockAllocator | None = None,
    prefixes: PrefixCache | None = None,
) -> ExecutionStage:
    started = time.perf_counter()
    results = batcher.run(requests)
    wall_ms = (time.perf_counter() - started) * 1_000
    correct = all(
        results[request_id].generated == expected
        for request_id, expected in references.items()
    )
    itls = [latency for result in results.values() for latency in result.itl_steps]
    metric = ExecutionStage(
        stage,
        name,
        correct,
        wall_ms,
        sum(request.max_new_tokens for request in requests),
        engine_steps=batcher.clock,
        mean_ttft_steps=sum(result.ttft_steps for result in results.values())
        / len(results),
        p95_itl_steps=_percentile([float(value) for value in itls], 0.95),
        peak_kv_blocks=allocator.peak_used_blocks if allocator is not None else None,
        prefix_hits=prefixes.hits if prefixes is not None else None,
    )
    if prefixes is not None:
        prefixes.close()
    return metric


def _pool(model: Transformer, *, blocks: int = 64) -> BlockAllocator:
    parameter = next(model.parameters())
    return BlockAllocator(
        num_layers=model.config.num_layers,
        num_blocks=blocks,
        block_size=2,
        num_kv_heads=model.config.num_kv_heads,
        head_dim=model.config.head_dim,
        device=parameter.device,
        dtype=parameter.dtype,
    )


def _paged_factory(pool: BlockAllocator):
    return lambda request: PagedKVCache(
        pool, max_tokens=request.prompt.numel() + request.max_new_tokens
    )


def _system_stage(
    stage: int,
    name: str,
    simulator: DisaggregatedSimulator,
    workload: list[WorkloadRequest],
) -> SystemStage:
    results = simulator.run(workload)
    return SystemStage(
        stage,
        name,
        summarize(workload, results),
        p95_transfer_ms=_percentile(
            [result.kv_transfer_ms for result in results.values()], 0.95
        ),
        p95_queue_ms=_percentile(
            [
                result.prefill_queue_ms
                + result.transfer_queue_ms
                + result.decode_queue_ms
                for result in results.values()
            ],
            0.95,
        ),
    )


def run_curriculum_scorecard(
    *, device: torch.device | str = "cpu"
) -> CurriculumScorecard:
    """Exercise every stage against one model and request shape."""
    device = torch.device(device)
    torch.manual_seed(42)
    model = (
        Transformer(
            ModelConfig(
                vocab_size=32,
                dim=32,
                num_layers=2,
                num_heads=4,
                num_kv_heads=1,
                max_seq_len=32,
            )
        )
        .eval()
        .to(device)
    )
    requests = _workload(device)
    references = _references(model, requests)
    output_tokens = sum(request.max_new_tokens for request in requests)

    execution = [
        _serial_stage(0, "naive", generate_naive, model, requests, references),
        _serial_stage(1, "kv_cache", generate_cached, model, requests, references),
    ]

    execution.append(
        _engine_stage(
            2,
            "continuous",
            ContinuousBatcher(model),
            requests,
            references,
        )
    )
    pool = _pool(model)
    execution.append(
        _engine_stage(
            3,
            "paged",
            ContinuousBatcher(model, cache_factory=_paged_factory(pool)),
            requests,
            references,
            allocator=pool,
        )
    )
    pool = _pool(model)
    prefixes = PrefixCache(pool)
    execution.append(
        _engine_stage(
            4,
            "prefix",
            ContinuousBatcher(
                model,
                cache_factory=_paged_factory(pool),
                prefix_cache=prefixes,
            ),
            requests,
            references,
            allocator=pool,
            prefixes=prefixes,
        )
    )
    pool = _pool(model)
    prefixes = PrefixCache(pool)
    execution.append(
        _engine_stage(
            5,
            "chunked",
            ContinuousBatcher(
                model,
                scheduler=FCFSScheduler(
                    max_batch_size=4,
                    max_batch_tokens=4,
                    prefill_chunk_size=2,
                ),
                cache_factory=_paged_factory(pool),
                prefix_cache=prefixes,
            ),
            requests,
            references,
            allocator=pool,
            prefixes=prefixes,
        )
    )

    started = time.perf_counter()
    speculative = [
        generate_speculative(
            model,
            model,
            request.prompt[None],
            max_new_tokens=request.max_new_tokens,
            draft_tokens=3,
        )
        for request in requests
    ]
    execution.append(
        ExecutionStage(
            6,
            "speculative",
            all(
                tuple(result.tokens[0, request.prompt.numel() :].tolist())
                == references[request.request_id]
                for request, result in zip(requests, speculative)
            ),
            (time.perf_counter() - started) * 1_000,
            output_tokens,
            target_calls=sum(result.target_calls for result in speculative),
        )
    )

    system_workload = [
        WorkloadRequest(
            request.request_id,
            request.prompt.numel(),
            request.max_new_tokens,
            arrival_ms=float(request.arrival_step),
            ttft_slo_ms=float(request.ttft_slo_steps)
            if request.ttft_slo_steps is not None
            else None,
            itl_slo_ms=float(request.itl_slo_steps)
            if request.itl_slo_steps is not None
            else None,
        )
        for request in requests
    ]
    arguments = dict(
        prefill_workers=[WorkerSpec("prefill-0", tokens_per_ms=2)],
        decode_workers=[WorkerSpec("decode-0", tokens_per_ms=1)],
        kv_bytes_per_token=512,
        transfer_bandwidth_bytes_per_ms=512,
        transfer_fixed_ms=0.5,
    )
    system = (
        _system_stage(
            7,
            "disaggregated_fcfs",
            DisaggregatedSimulator(**arguments),
            system_workload,
        ),
        _system_stage(
            8,
            "slo_aware",
            DisaggregatedSimulator(**arguments, slo_aware=True),
            system_workload,
        ),
    )
    return CurriculumScorecard(tuple(execution), system)


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog=prog, description="Run the complete microServe curriculum scorecard."
    )
    parser.parse_args(argv)
    scorecard = run_curriculum_scorecard()
    print("model execution (one model, one request workload)")
    print("stage mechanism       correct  wall ms  tok/s  steps  TTFT  ITL  blocks")
    for item in scorecard.execution:
        print(
            f"{item.stage:>5} {item.name:<15} {str(item.correct):<7} "
            f"{item.wall_ms:>8.2f} {item.output_tokens_per_second:>6.0f} "
            f"{_optional(item.engine_steps):>6} "
            f"{_optional(item.mean_ttft_steps):>5} "
            f"{_optional(item.p95_itl_steps):>4} "
            f"{_optional(item.peak_kv_blocks):>7}"
        )
    print("\nsystem simulation (same prompt/output/arrival shape)")
    print("stage mechanism          p50 TTFT  p95 TTFT  p95 ITL  SLO%  queue  transfer")
    for item in scorecard.system:
        summary = item.summary
        print(
            f"{item.stage:>5} {item.name:<18} {summary.p50_ttft_ms:>8.2f} "
            f"{summary.p95_ttft_ms:>9.2f} {summary.p95_itl_ms:>8.2f} "
            f"{summary.slo_attainment * 100:>5.0f} "
            f"{item.p95_queue_ms:>6.2f} {item.p95_transfer_ms:>9.2f}"
        )


if __name__ == "__main__":
    main()
