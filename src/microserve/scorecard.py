"""Compare serving-system configurations on one deterministic workload."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

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
from microserve.disagg import (
    DisaggregatedSimulator,
    LatencySummary,
    WorkerSpec,
    WorkloadRequest,
    summarize,
)
from microserve.memory import (
    MemoryBudgetError,
    estimate_memory,
    release_device_memory,
    require_memory_budget,
)
from microserve.profiler import (
    ServiceProfile,
    profile_model,
    synthetic_service_profile,
)


DEFAULT_LINK_GBPS = 25.0


@dataclass(frozen=True)
class Configuration:
    """The topology and policy varied by one scorecard row."""

    name: str
    policy: str
    prefill_workers: int
    decode_workers: int
    link_scale: float = 1.0

    @property
    def slo_aware(self) -> bool:
        return self.policy == "SLO"

    @property
    def workers(self) -> str:
        return f"{self.prefill_workers}P/{self.decode_workers}D"


@dataclass(frozen=True)
class ConfigurationResult:
    configuration: Configuration
    summary: LatencySummary
    p95_transfer_ms: float
    p95_queue_ms: float


@dataclass(frozen=True)
class SystemScorecard:
    profile: ServiceProfile
    workload: tuple[WorkloadRequest, ...]
    configurations: tuple[ConfigurationResult, ...]
    link_gbps: float
    transfer_fixed_ms: float

    # Preserve the old read-only result spelling for callers while making the
    # CLI and primary API configuration-centric.
    @property
    def system(self) -> tuple[ConfigurationResult, ...]:
        return self.configurations

    @property
    def execution(self) -> tuple[()]:
        return ()


# Compatibility name retained for callers of the original scorecard API.
CurriculumScorecard = SystemScorecard


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _workload(profile: ServiceProfile) -> tuple[WorkloadRequest, ...]:
    """Four arrival waves, each mixing urgent, chat, and long requests."""
    rows = (
        ("chat_0", 8, 10, 0, 18),
        ("long_0", 32, 14, 0, 45),
        ("urgent_0", 4, 4, 0, 7),
        ("chat_1", 10, 10, 4, 20),
        ("long_1", 40, 14, 4, 50),
        ("urgent_1", 5, 4, 4, 8),
        ("chat_2", 12, 10, 8, 22),
        ("long_2", 48, 14, 8, 55),
        ("urgent_2", 6, 4, 8, 9),
        ("chat_3", 14, 10, 12, 24),
        ("long_3", 56, 14, 12, 60),
        ("urgent_3", 4, 4, 12, 8),
    )
    # Arrival spacing and latency objectives track the measured decode latency,
    # so the same workload remains meaningful on a laptop or accelerator.
    time_scale = profile.decode.median_ms
    return tuple(
        WorkloadRequest(
            request_id,
            prompt_tokens,
            output_tokens,
            arrival_ms=arrival_ms * time_scale,
            ttft_slo_ms=ttft_slo_ms * time_scale,
            itl_slo_ms=profile.decode.p95_ms * 1.10,
        )
        for request_id, prompt_tokens, output_tokens, arrival_ms, ttft_slo_ms in rows
    )


def _configurations() -> tuple[Configuration, ...]:
    """Vary one likely bottleneck at a time, then scale the whole system."""
    return (
        Configuration("baseline", "FCFS", 1, 1),
        Configuration("slo_policy", "SLO", 1, 1),
        Configuration("prefill_heavy", "SLO", 2, 1),
        Configuration("decode_heavy", "SLO", 1, 2),
        Configuration("balanced_slow_link", "SLO", 2, 2, 0.25),
        Configuration("balanced", "SLO", 2, 2),
        Configuration("decode_scaled", "SLO", 2, 4),
        Configuration("scaled_fast_link", "SLO", 4, 4, 4.0),
    )


def _workers(
    prefix: str,
    count: int,
    tokens_per_ms: float,
    fixed_ms: float = 0.0,
) -> list[WorkerSpec]:
    return [
        WorkerSpec(f"{prefix}-{index}", tokens_per_ms, fixed_ms)
        for index in range(count)
    ]


def _bytes_per_ms(gigabits_per_second: float) -> float:
    return gigabits_per_second * 125_000


def _run_configuration(
    configuration: Configuration,
    workload: tuple[WorkloadRequest, ...],
    profile: ServiceProfile,
    *,
    link_gbps: float,
    transfer_fixed_ms: float,
) -> ConfigurationResult:
    simulator = DisaggregatedSimulator(
        prefill_workers=_workers(
            "prefill",
            configuration.prefill_workers,
            profile.prefill_tokens_per_ms,
            profile.prefill_fixed_ms,
        ),
        decode_workers=_workers(
            "decode", configuration.decode_workers, profile.decode_tokens_per_ms
        ),
        kv_bytes_per_token=profile.kv_bytes_per_token,
        transfer_bandwidth_bytes_per_ms=(
            _bytes_per_ms(link_gbps) * configuration.link_scale
        ),
        transfer_fixed_ms=transfer_fixed_ms,
        slo_aware=configuration.slo_aware,
    )
    results = simulator.run(list(workload))
    return ConfigurationResult(
        configuration=configuration,
        summary=summarize(list(workload), results),
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


def run_system_scorecard(
    profile: ServiceProfile,
    *,
    link_gbps: float = 0.065536,
    transfer_fixed_ms: float = 0.5,
) -> SystemScorecard:
    workload = _workload(profile)
    configurations = tuple(
        _run_configuration(
            configuration,
            workload,
            profile,
            link_gbps=link_gbps,
            transfer_fixed_ms=transfer_fixed_ms,
        )
        for configuration in _configurations()
    )
    return SystemScorecard(
        profile,
        workload,
        configurations,
        link_gbps,
        transfer_fixed_ms,
    )


def run_curriculum_scorecard(*, device: object | None = None) -> SystemScorecard:
    """Compatibility wrapper for the original public function."""
    del device
    return run_system_scorecard(synthetic_service_profile())


def _link(scale: float) -> str:
    return f"{scale:g}x"


def _measured_table(profile: ServiceProfile) -> Table:
    table = Table(title="Measured model execution — synchronized wall time")
    table.add_column("operation", style="bold")
    table.add_column("shape")
    table.add_column("median", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("tok/s", justify="right")
    table.add_column("samples", justify="right")
    for measurement in (*profile.prefill, profile.decode, profile.generation):
        table.add_row(
            measurement.operation,
            measurement.shape,
            f"{measurement.median_ms:.2f} ms",
            f"{measurement.p95_ms:.2f} ms",
            f"{measurement.tokens_per_second:,.1f}",
            str(measurement.samples),
        )
    return table


def _projection_table(scorecard: SystemScorecard) -> str:
    lines = [
        "Projected system configurations — calibrated, not executed",
        "configuration       policy workers link  p50 TTFT  p95 TTFT  "
        "p95 ITL  SLO%  queue  transfer  tok/s  gain",
    ]
    baseline = scorecard.configurations[0].summary.output_tokens_per_second
    for result in scorecard.configurations:
        config = result.configuration
        summary = result.summary
        lines.append(
            f"{config.name:<19} {config.policy:<6} {config.workers:>7} "
            f"{_link(config.link_scale):>4} "
            f"{summary.p50_ttft_ms:>9.2f} {summary.p95_ttft_ms:>9.2f} "
            f"{summary.p95_itl_ms:>8.2f} {summary.slo_attainment * 100:>5.0f} "
            f"{result.p95_queue_ms:>6.2f} {result.p95_transfer_ms:>9.2f} "
            f"{summary.output_tokens_per_second:>6.0f} "
            f"{summary.output_tokens_per_second / baseline:>5.2f}x"
        )
    return "\n".join(lines)


def _parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Measure a real model, then project serving configurations.",
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
    parser.add_argument("--link-gbps", type=float, default=DEFAULT_LINK_GBPS)
    parser.add_argument("--transfer-fixed-ms", type=float, default=0.1)
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
    if args.link_gbps <= 0 or args.transfer_fixed_ms < 0:
        parser.error("link bandwidth must be positive and fixed latency non-negative")
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
                f"profile workload exceeds model context "
                f"{checkpoint.model.max_seq_len}"
            )
        estimate = estimate_memory(
            checkpoint.model,
            method="kv_cache",
            batch_size=1,
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
            batch_size=1,
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
        with console.status("Measuring synchronized model execution…"):
            profile = profile_model(
                model,
                model_id=args.model,
                prompt_lengths=args.prompt_lengths,
                decode_steps=args.decode_steps,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        scorecard = run_system_scorecard(
            profile,
            link_gbps=args.link_gbps,
            transfer_fixed_ms=args.transfer_fixed_ms,
        )

        console.print(_measured_table(profile))
        prefill_fit = (
            f"constant {profile.prefill_fixed_ms:.2f} ms"
            if math.isinf(profile.prefill_tokens_per_ms)
            else (
                f"{profile.prefill_fixed_ms:.2f} ms + tokens / "
                f"{profile.prefill_tokens_per_ms:.3f} tok/ms"
            )
        )
        console.print(
            f"Fitted prefill: {prefill_fit}  •  "
            f"decode: {profile.decode.tokens_per_second:.1f} tok/s  •  "
            f"KV: {profile.kv_bytes_per_token:,} bytes/token"
        )
        console.print(_projection_table(scorecard), markup=False, soft_wrap=True)
        console.print(
            "[dim]Measured table: actual model calls on this device. "
            "Projection table: virtual replicas calibrated from those calls; "
            f"network is assumed at {args.link_gbps:g} Gbit/s × link.[/dim]"
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
