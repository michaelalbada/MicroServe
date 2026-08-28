"""A fast, network-free demonstration of the KV-cache crossover."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from microserve.benchmark import measure
from microserve.checkpoint import resolve_dtype
from microserve.cli_ui import (
    default_device,
    render_memory_plan,
    render_run_summary,
    validate_device,
)
from microserve.generate import generate_cached, generate_naive
from microserve.memory import estimate_memory, release_device_memory
from microserve.model import ModelConfig, Transformer


def _parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Show when KV reuse overtakes full-sequence recomputation.",
    )
    parser.add_argument("--device", default=str(default_device()))
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    parser = _parser(prog=prog)
    args = parser.parse_args(argv)
    console = Console()
    device = torch.device(args.device)
    if unavailable := validate_device(device):
        parser.error(unavailable)
    dtype = resolve_dtype(args.dtype, device)
    torch.manual_seed(0)

    batch_size = 4
    new_tokens = 32
    contexts = (16, 64, 128)
    config = ModelConfig(
        vocab_size=256,
        dim=128,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        hidden_dim=512,
        max_seq_len=max(contexts) + new_tokens,
    )
    render_run_summary(
        console,
        source="network-free quickstart model (random weights)",
        config=config,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        prompt_tokens=max(contexts),
        new_tokens=new_tokens,
    )
    estimates = [
        estimate_memory(
            config,
            method=method,
            batch_size=batch_size,
            prompt_tokens=max(contexts),
            new_tokens=new_tokens,
            dtype=dtype,
            device=device,
        )
        for method in ("naive", "kv_cache")
    ]
    render_memory_plan(console, estimates)

    with console.status("Warming up both generation paths…"):
        model = Transformer(config).eval().to(device=device, dtype=dtype)
        warmup_prompt = torch.randint(config.vocab_size, (batch_size, 4), device=device)
        generate_naive(model, warmup_prompt, max_new_tokens=1)
        generate_cached(model, warmup_prompt, max_new_tokens=1)

    rows = []
    total = len(contexts) * 2 * batch_size * new_tokens
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=args.no_progress,
    ) as progress:
        task = progress.add_task("context sweep", total=total)
        for context in contexts:
            prompt = torch.randint(
                config.vocab_size, (batch_size, context), device=device
            )
            naive = measure(
                generate_naive,
                model,
                prompt,
                max_new_tokens=new_tokens,
                warmup=0,
                repeats=1,
                progress=lambda advance: progress.update(task, advance=advance),
            )
            cached = measure(
                generate_cached,
                model,
                prompt,
                max_new_tokens=new_tokens,
                warmup=0,
                repeats=1,
                progress=lambda advance: progress.update(task, advance=advance),
            )
            torch.testing.assert_close(cached.output, naive.output)
            rows.append((context, naive, cached))

    table = Table(title="Why serving needs a KV cache")
    table.add_column("prompt", justify="right")
    table.add_column("naive", justify="right")
    table.add_column("KV cache", justify="right")
    table.add_column("speedup", justify="right", style="bold green")
    table.add_column("tokens", justify="center")
    for context, naive, cached in rows:
        table.add_row(
            str(context),
            f"{naive.median_seconds * 1_000:.1f} ms",
            f"{cached.median_seconds * 1_000:.1f} ms",
            f"{naive.median_seconds / cached.median_seconds:.2f}×",
            "identical",
        )
    console.print(table)
    console.print(
        Panel(
            "Naive generation replays the growing prefix for every token. The "
            "cached path prefills once and then processes one new token per step. "
            "The exact crossover is device-dependent; the scaling trend is the lesson.\n\n"
            'Next: [bold]microserve generate "The capital of France is"[/bold]\n'
            "Deeper: [bold]microserve scorecard[/bold]",
            title="What you just measured",
            border_style="green",
        )
    )
    release_device_memory(device)


if __name__ == "__main__":
    main()
