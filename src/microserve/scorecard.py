"""Execute serving-system configurations on one deterministic workload."""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from microserve.allocator import BlockAllocator, PagedKVCache
from microserve.batcher import ContinuousBatcher, Request
from microserve.checkpoint import (
    DEFAULT_MODEL,
    fetch_snapshot,
    load_llama_weights,
    read_checkpoint_config,
    resolve_dtype,
)
from microserve.cli_ui import (
    default_device,
    is_out_of_memory,
    render_memory_plan,
    render_memory_refusal,
    render_oom,
    render_run_summary,
    validate_device,
)
from microserve.memory import (
    MemoryBudgetError,
    estimate_memory,
    release_device_memory,
    require_memory_budget,
)
from microserve.scheduler import FCFSScheduler, SLOAwareScheduler


@dataclass(frozen=True)
class ExecutionConfiguration:
    """One serving design that can be executed by this process."""

    name: str
    cache: str
    policy: str
    sequential: bool = False
    chunked: bool = False


@dataclass(frozen=True)
class ExecutionMeasurement:
    """Observed wall-clock metrics for one executable configuration."""

    configuration: ExecutionConfiguration
    correct: bool
    median_wall_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    p95_itl_ms: float
    output_tokens_per_second: float
    peak_blocks: int | None


@dataclass(frozen=True)
class _ExecutionRun:
    wall_ms: float
    ttft_ms: tuple[float, ...]
    itl_ms: tuple[float, ...]
    outputs: dict[str, tuple[int, ...]]
    peak_blocks: int | None


EXECUTION_CONFIGURATIONS = (
    ExecutionConfiguration("sequential", "contiguous", "FCFS", sequential=True),
    ExecutionConfiguration("continuous", "contiguous", "FCFS"),
    ExecutionConfiguration("paged", "paged", "FCFS"),
    ExecutionConfiguration("chunked", "paged", "FCFS", chunked=True),
    ExecutionConfiguration("slo_aware", "paged", "SLO", chunked=True),
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _runtime_requests(
    model: torch.nn.Module,
    *,
    prompt_lengths: Sequence[int],
    decode_steps: int,
) -> tuple[Request, ...]:
    parameter = next(model.parameters())
    generator = torch.Generator()
    generator.manual_seed(0)
    shortest_first = {
        index: rank + 1
        for rank, index in enumerate(
            sorted(range(len(prompt_lengths)), key=lambda item: prompt_lengths[item])
        )
    }
    return tuple(
        Request(
            request_id=f"request_{index}_prompt_{length}",
            prompt=torch.randint(
                model.config.vocab_size,
                (length,),
                generator=generator,
            ).to(parameter.device),
            max_new_tokens=decode_steps,
            ttft_slo_steps=shortest_first[index],
            itl_slo_steps=1,
        )
        for index, length in enumerate(prompt_lengths)
    )


def _runtime_scheduler(
    configuration: ExecutionConfiguration,
    requests: Sequence[Request],
    *,
    chunk_size: int,
) -> FCFSScheduler:
    max_batch_size = 1 if configuration.sequential else len(requests)
    if configuration.chunked:
        max_batch_tokens = max(len(requests), len(requests) * chunk_size)
        kwargs = {"prefill_chunk_size": chunk_size}
    else:
        max_batch_tokens = max(
            len(requests), sum(request.prompt.numel() for request in requests)
        )
        kwargs = {}
    scheduler_type = (
        SLOAwareScheduler if configuration.policy == "SLO" else FCFSScheduler
    )
    return scheduler_type(
        max_batch_size=max_batch_size,
        max_batch_tokens=max_batch_tokens,
        **kwargs,
    )


def _run_execution_once(
    model: torch.nn.Module,
    requests: Sequence[Request],
    configuration: ExecutionConfiguration,
    *,
    block_size: int,
    chunk_size: int,
) -> _ExecutionRun:
    parameter = next(model.parameters())
    device = parameter.device
    allocator = None
    cache_factory = None
    if configuration.cache == "paged":
        capacities = [
            request.prompt.numel() + request.max_new_tokens for request in requests
        ]
        num_blocks = sum(math.ceil(capacity / block_size) for capacity in capacities)
        allocator = BlockAllocator(
            num_layers=model.config.num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=model.config.num_kv_heads,
            head_dim=model.config.head_dim,
            device=device,
            dtype=parameter.dtype,
        )

        def cache_factory(request: Request) -> PagedKVCache:
            return PagedKVCache(
                allocator,
                max_tokens=request.prompt.numel() + request.max_new_tokens,
            )

    token_times: dict[str, list[float]] = defaultdict(list)
    started = 0.0

    def record_token(request: Request, token: int) -> None:
        del token
        token_times[request.request_id].append((time.perf_counter() - started) * 1_000)

    batcher = ContinuousBatcher(
        model,
        scheduler=_runtime_scheduler(
            configuration,
            requests,
            chunk_size=chunk_size,
        ),
        cache_factory=cache_factory,
        token_callback=record_token,
    )
    _synchronize(device)
    started = time.perf_counter()
    results = batcher.run(list(requests))
    _synchronize(device)
    wall_ms = (time.perf_counter() - started) * 1_000
    ttft = tuple(token_times[request.request_id][0] for request in requests)
    itl = tuple(
        right - left
        for request in requests
        for left, right in zip(
            token_times[request.request_id], token_times[request.request_id][1:]
        )
    )
    outputs = {request_id: result.generated for request_id, result in results.items()}
    return _ExecutionRun(
        wall_ms,
        ttft,
        itl,
        outputs,
        allocator.peak_used_blocks if allocator is not None else None,
    )


def measure_execution_configuration(
    model: torch.nn.Module,
    requests: Sequence[Request],
    configuration: ExecutionConfiguration,
    *,
    reference: dict[str, tuple[int, ...]] | None,
    warmup: int,
    repeats: int,
    block_size: int = 16,
    chunk_size: int = 128,
) -> tuple[ExecutionMeasurement, dict[str, tuple[int, ...]]]:
    """Execute one configuration and aggregate only observed timings."""
    runs = [
        _run_execution_once(
            model,
            requests,
            configuration,
            block_size=block_size,
            chunk_size=chunk_size,
        )
        for _ in range(warmup + repeats)
    ][warmup:]
    outputs = runs[-1].outputs
    expected = outputs if reference is None else reference
    wall_ms = statistics.median(run.wall_ms for run in runs)
    ttft = [value for run in runs for value in run.ttft_ms]
    itl = [value for run in runs for value in run.itl_ms]
    output_tokens = sum(request.max_new_tokens for request in requests)
    measurement = ExecutionMeasurement(
        configuration=configuration,
        correct=outputs == expected,
        median_wall_ms=wall_ms,
        p50_ttft_ms=_percentile(ttft, 0.50),
        p95_ttft_ms=_percentile(ttft, 0.95),
        p95_itl_ms=_percentile(itl, 0.95),
        output_tokens_per_second=1_000 * output_tokens / wall_ms,
        peak_blocks=max(
            (run.peak_blocks for run in runs if run.peak_blocks is not None),
            default=None,
        ),
    )
    return measurement, outputs


def _execution_table(measurements: Sequence[ExecutionMeasurement]) -> Table:
    table = Table(title="Measured serving configurations — synchronized wall time")
    table.add_column("configuration", style="bold")
    table.add_column("cache")
    table.add_column("policy")
    table.add_column("correct", justify="center")
    table.add_column("wall", justify="right")
    table.add_column("p50 TTFT", justify="right")
    table.add_column("p95 TTFT", justify="right")
    table.add_column("p95 ITL", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("blocks", justify="right")
    for item in measurements:
        config = item.configuration
        table.add_row(
            config.name,
            config.cache,
            config.policy,
            str(item.correct),
            f"{item.median_wall_ms:.2f} ms",
            f"{item.p50_ttft_ms:.2f} ms",
            f"{item.p95_ttft_ms:.2f} ms",
            f"{item.p95_itl_ms:.2f} ms",
            f"{item.output_tokens_per_second:.1f}",
            "-" if item.peak_blocks is None else str(item.peak_blocks),
        )
    return table


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Execute serving configurations with a real model.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default=str(default_device()))
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=(8, 32, 56))
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=tuple(config.name for config in EXECUTION_CONFIGURATIONS),
        default=tuple(config.name for config in EXECUTION_CONFIGURATIONS),
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    # Accepted for compatibility with the former projection-only scorecard.
    parser.add_argument("--link-gbps", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--transfer-fixed-ms", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--memory-fraction", type=float, default=0.75)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    parser = _parser(prog=prog)
    args = parser.parse_args(argv)
    if not args.prompt_lengths or min(args.prompt_lengths) < 1:
        parser.error("prompt lengths must be positive")
    if args.decode_steps < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("decode steps/repeats must be positive and warmup non-negative")
    if args.block_size < 1 or args.chunk_size < 1:
        parser.error("block size and chunk size must be positive")
    if not 0 < args.memory_fraction <= 1:
        parser.error("memory fraction must be in (0, 1]")

    console = Console()
    device = torch.device(args.device)
    if unavailable := validate_device(device):
        parser.error(unavailable)

    try:
        console.print(f"[bold]Resolving model[/bold] {args.model}")
        snapshot = fetch_snapshot(
            args.model,
            revision=args.revision,
            cache_dir=args.cache_dir,
            local_files_only=args.offline,
        )
        checkpoint = read_checkpoint_config(snapshot)
        dtype = resolve_dtype(args.dtype, device, checkpoint.torch_dtype)
        longest_prompt = max(args.prompt_lengths)
        if longest_prompt + args.decode_steps > checkpoint.model.max_seq_len:
            parser.error(
                f"request workload exceeds model context {checkpoint.model.max_seq_len}"
            )
        estimate = estimate_memory(
            checkpoint.model,
            method="kv_cache",
            batch_size=len(args.prompt_lengths),
            prompt_tokens=longest_prompt,
            new_tokens=args.decode_steps,
            dtype=dtype,
            device=device,
            memory_fraction=args.memory_fraction,
        )
        render_run_summary(
            console,
            source=f"{args.model} (real safetensors)",
            config=checkpoint.model,
            device=device,
            dtype=dtype,
            batch_size=len(args.prompt_lengths),
            prompt_tokens=longest_prompt,
            new_tokens=args.decode_steps,
        )
        render_memory_plan(console, [estimate], forced=args.force)
        try:
            require_memory_budget([estimate], force=args.force)
        except MemoryBudgetError:
            render_memory_refusal(console)
            raise SystemExit(2) from None

        with console.status("Loading weights onto the device…"):
            model, _ = load_llama_weights(snapshot, device=device, dtype=args.dtype)
        requests = _runtime_requests(
            model,
            prompt_lengths=args.prompt_lengths,
            decode_steps=args.decode_steps,
        )
        selected = [
            config
            for config in EXECUTION_CONFIGURATIONS
            if config.name in args.configurations
        ]
        output_tokens_per_run = len(requests) * args.decode_steps
        total_output_tokens = (
            output_tokens_per_run * (args.warmup + args.repeats) * len(selected)
        )
        console.print(
            f"[dim]Actual workload: {len(requests)} concurrent requests × "
            f"{args.decode_steps} output tokens; {args.warmup + args.repeats} "
            f"runs/configuration × {len(selected)} configurations = "
            f"{total_output_tokens:,} generated tokens.[/dim]"
        )
        measurements = []
        reference = None
        for configuration in selected:
            with console.status(
                f"Executing [bold]{configuration.name}[/bold] "
                f"({args.warmup} warmup + {args.repeats} measured)…"
            ):
                measurement, outputs = measure_execution_configuration(
                    model,
                    requests,
                    configuration,
                    reference=reference,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    block_size=args.block_size,
                    chunk_size=args.chunk_size,
                )
            if reference is None:
                reference = outputs
            measurements.append(measurement)
            console.print(
                f"  [green]✓[/green] {configuration.name}: "
                f"{measurement.median_wall_ms:.2f} ms, "
                f"{measurement.output_tokens_per_second:.1f} tok/s"
            )

        console.print(_execution_table(measurements))
        console.print(
            "[dim]Every row executed the complete workload with the loaded model. "
            "TTFT and ITL are observed token emission times; no service-time fit, "
            "virtual replica, or network model is used.[/dim]"
        )
    except SystemExit:
        raise
    except KeyboardInterrupt:
        release_device_memory(device)
        console.print("\n[yellow]Scorecard cancelled; device cache released.[/yellow]")
        raise SystemExit(130) from None
    except RuntimeError as error:
        release_device_memory(device)
        if is_out_of_memory(error):
            render_oom(console, error)
            raise SystemExit(2) from error
        raise
    except Exception as error:
        release_device_memory(device)
        console.print_exception(show_locals=False)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
