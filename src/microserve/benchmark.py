"""Benchmark naive and KV-cached generation with visible memory/progress."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from microserve.checkpoint import (
    fetch_snapshot,
    load_llama_weights,
    load_tokenizer,
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
from microserve.generate import generate_cached, generate_naive
from microserve.memory import (
    MemoryBudgetError,
    estimate_memory,
    release_device_memory,
    require_memory_budget,
)
from microserve.model import ModelConfig, Transformer


Generate = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class Measurement:
    median_seconds: float
    tokens_per_second: float
    output: torch.Tensor = field(repr=False, compare=False)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure(
    generate: Generate,
    model: Transformer,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    warmup: int,
    repeats: int,
    progress: Callable[[int], None] | None = None,
) -> Measurement:
    """Measure steady-state end-to-end generation for a fixed batch."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    output = prompt
    for _ in range(warmup):
        output = generate(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            progress=progress,
        )
    _synchronize(prompt.device)

    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = generate(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            progress=progress,
        )
        _synchronize(prompt.device)
        durations.append(time.perf_counter() - started)

    median_seconds = statistics.median(durations)
    generated_tokens = prompt.size(0) * max_new_tokens
    return Measurement(
        median_seconds=median_seconds,
        tokens_per_second=generated_tokens / median_seconds,
        output=output,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare naive and KV-cached autoregressive generation."
    )
    parser.add_argument("--device", default=str(default_device()))
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--model",
        help="Hugging Face Llama model ID; omit for explicit random toy weights.",
    )
    parser.add_argument("--revision")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--prompt", help="Text prompt for a real model; otherwise use random tokens."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("naive", "kv_cache"),
        default=("naive", "kv_cache"),
    )
    parser.add_argument("--memory-fraction", type=float, default=0.75)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if min(args.batch_size, args.prompt_tokens, args.repeats) < 1:
        parser.error("batch size, prompt tokens, and repeats must be positive")
    if args.new_tokens < 1 or args.warmup < 0:
        parser.error("new tokens must be positive and warmup non-negative")
    if not 0 < args.memory_fraction <= 1:
        parser.error("memory fraction must be in (0, 1]")
    if args.prompt and not args.model:
        parser.error("--prompt requires --model so text can be tokenized")


def _random_config(max_seq_len: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        dim=256,
        num_layers=6,
        num_heads=8,
        num_kv_heads=2,
        max_seq_len=max_seq_len,
    )


def _results_table(results: dict[str, Measurement], *, baseline: float | None) -> Table:
    table = Table(title="Generation benchmark")
    table.add_column("method", style="bold")
    table.add_column("median", justify="right")
    table.add_column("generated tok/s", justify="right")
    table.add_column("speedup", justify="right")
    for name, result in results.items():
        speedup = baseline / result.median_seconds if baseline is not None else None
        table.add_row(
            name,
            f"{result.median_seconds * 1_000:.2f} ms",
            f"{result.tokens_per_second:,.2f}",
            f"{speedup:.2f}×" if speedup is not None else "—",
        )
    return table


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    _validate(args, parser)
    console = Console()
    device = torch.device(args.device)
    if unavailable := validate_device(device):
        parser.error(unavailable)
    torch.manual_seed(0)

    try:
        snapshot = None
        tokenizer = None
        if args.model:
            console.print(f"[bold]Resolving model[/bold] {args.model}")
            snapshot = fetch_snapshot(
                args.model,
                revision=args.revision,
                cache_dir=args.cache_dir,
                local_files_only=args.offline,
            )
            checkpoint = read_checkpoint_config(snapshot)
            config = checkpoint.model
            dtype = resolve_dtype(args.dtype, device, checkpoint.torch_dtype)
            source = f"{args.model} (real safetensors)"
            if args.prompt:
                tokenizer = load_tokenizer(snapshot, checkpoint)
                prompt_ids = tokenizer.encode(args.prompt)
                args.prompt_tokens = len(prompt_ids)
        else:
            config = _random_config(args.prompt_tokens + args.new_tokens)
            dtype = resolve_dtype(args.dtype, device)
            source = "random toy weights (no model download)"

        if args.prompt_tokens + args.new_tokens > config.max_seq_len:
            parser.error(f"prompt + decode exceeds model context {config.max_seq_len}")
        estimates = [
            estimate_memory(
                config,
                method=method,
                batch_size=args.batch_size,
                prompt_tokens=args.prompt_tokens,
                new_tokens=args.new_tokens,
                dtype=dtype,
                device=device,
                memory_fraction=args.memory_fraction,
            )
            for method in dict.fromkeys(args.methods)
        ]
        render_run_summary(
            console,
            source=source,
            config=config,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            prompt_tokens=args.prompt_tokens,
            new_tokens=args.new_tokens,
        )
        render_memory_plan(console, estimates, forced=args.force)
        try:
            require_memory_budget(estimates, force=args.force)
        except MemoryBudgetError:
            render_memory_refusal(console)
            raise SystemExit(2) from None

        with console.status("Loading weights onto the device…"):
            if snapshot is not None:
                model, _ = load_llama_weights(snapshot, device=device, dtype=args.dtype)
            else:
                model = Transformer(config).eval().to(device=device, dtype=dtype)

        if tokenizer is not None:
            base_prompt = torch.tensor(prompt_ids, device=device, dtype=torch.long)
            prompt = base_prompt[None, :].repeat(args.batch_size, 1)
        else:
            prompt = torch.randint(
                config.vocab_size,
                (args.batch_size, args.prompt_tokens),
                device=device,
            )

        methods = {
            "naive": generate_naive,
            "kv_cache": generate_cached,
        }
        results: dict[str, Measurement] = {}
        total_per_method = (
            args.batch_size * args.new_tokens * (args.warmup + args.repeats)
        )
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            disable=args.no_progress,
        ) as progress:
            for name in dict.fromkeys(args.methods):
                task = progress.add_task(name, total=total_per_method)
                results[name] = measure(
                    methods[name],
                    model,
                    prompt,
                    max_new_tokens=args.new_tokens,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    progress=lambda advance, task=task: progress.update(
                        task, advance=advance
                    ),
                )
                release_device_memory(device)

        if len(results) > 1:
            outputs = iter(results.values())
            reference = next(outputs).output
            for result in outputs:
                torch.testing.assert_close(result.output, reference)
            console.print("[green]✓ methods produced identical tokens[/green]")
        baseline = results["naive"].median_seconds if "naive" in results else None
        console.print(_results_table(results, baseline=baseline))
    except SystemExit:
        raise
    except KeyboardInterrupt:
        release_device_memory(device)
        console.print("\n[yellow]Benchmark cancelled; device cache released.[/yellow]")
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
