"""Shared Rich presentation helpers for microServe commands."""

from __future__ import annotations

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from microserve.memory import MemoryEstimate, format_bytes, parameter_count
from microserve.model import ModelConfig


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_device(device: torch.device) -> str | None:
    if device.type == "mps" and not torch.backends.mps.is_available():
        return "MPS was requested but is not available in this process."
    if device.type == "cuda" and not torch.cuda.is_available():
        return "CUDA was requested but is not available in this process."
    return None


def render_run_summary(
    console: Console,
    *,
    source: str,
    config: ModelConfig,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    prompt_tokens: int,
    new_tokens: int,
) -> None:
    parameters = parameter_count(config)
    console.print(
        Panel.fit(
            f"[bold]{source}[/bold]\n"
            f"{parameters / 1_000_000:.1f}M parameters  •  "
            f"{config.num_layers} layers × width {config.dim}\n"
            f"device [cyan]{device}[/cyan]  •  dtype [cyan]{dtype}[/cyan]\n"
            f"batch {batch_size}  •  prompt {prompt_tokens}  •  decode {new_tokens}",
            title="microServe",
            border_style="cyan",
        )
    )


def render_memory_plan(
    console: Console, estimates: list[MemoryEstimate], *, forced: bool = False
) -> None:
    table = Table(title="Memory preflight", show_lines=False)
    table.add_column("method", style="bold")
    table.add_column("model", justify="right")
    table.add_column("KV", justify="right")
    table.add_column("attention", justify="right")
    table.add_column("MPS churn", justify="right")
    table.add_column("est. peak", justify="right")
    table.add_column("safe limit", justify="right")
    table.add_column("status", justify="center")
    for item in estimates:
        if item.fits:
            status = "[green]ready[/green]"
        elif forced:
            status = "[yellow]forced[/yellow]"
        else:
            status = "[red]refuse[/red]"
        table.add_row(
            item.method,
            format_bytes(item.model_bytes),
            format_bytes(item.kv_cache_bytes),
            format_bytes(item.attention_bytes),
            format_bytes(item.allocator_churn_bytes),
            format_bytes(item.estimated_peak_bytes),
            format_bytes(item.limit_bytes),
            status,
        )
    console.print(table)


def render_memory_refusal(console: Console) -> None:
    console.print(
        Panel(
            "The requested run is likely to exhaust the device. Reduce "
            "[bold]--batch-size[/bold], [bold]--prompt-tokens[/bold], or "
            "[bold]--new-tokens[/bold]; skip quadratic recomputation with "
            "[bold]--methods kv_cache[/bold]; or use [bold]--force[/bold] "
            "only when you understand the estimate.",
            title="Run refused before allocation",
            border_style="red",
        )
    )


def is_out_of_memory(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "invalid buffer size" in message


def render_oom(console: Console, error: BaseException) -> None:
    console.print(
        Panel(
            f"{error}\n\n"
            "Device memory was released. Try a smaller workload or rerun only "
            "the cached path with [bold]--methods kv_cache[/bold].",
            title="Device ran out of memory",
            border_style="red",
        )
    )
