"""Generate text with a real checkpoint using microServe's own runtime."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

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

from microserve.checkpoint import (
    DEFAULT_MODEL,
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
from microserve.generate import generate_cached
from microserve.memory import (
    MemoryBudgetError,
    estimate_memory,
    release_device_memory,
    require_memory_budget,
)


def _parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Fetch and run a real Llama-format model with microServe.",
    )
    parser.add_argument("prompt", help="Text to complete.")
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
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--memory-fraction", type=float, default=0.75)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--show-token-ids", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> None:
    parser = _parser(prog=prog)
    args = parser.parse_args(argv)
    if args.new_tokens < 1:
        parser.error("new tokens must be positive")
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
        tokenizer = load_tokenizer(snapshot, checkpoint)
        prompt_ids = tokenizer.encode(args.prompt)
        if not prompt_ids:
            parser.error("the tokenizer produced an empty prompt")
        if len(prompt_ids) + args.new_tokens > checkpoint.model.max_seq_len:
            parser.error(
                f"prompt + decode exceeds model context {checkpoint.model.max_seq_len}"
            )
        dtype = resolve_dtype(args.dtype, device, checkpoint.torch_dtype)
        estimate = estimate_memory(
            checkpoint.model,
            method="kv_cache",
            batch_size=1,
            prompt_tokens=len(prompt_ids),
            new_tokens=args.new_tokens,
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
            prompt_tokens=len(prompt_ids),
            new_tokens=args.new_tokens,
        )
        render_memory_plan(console, [estimate], forced=args.force)
        try:
            require_memory_budget([estimate], force=args.force)
        except MemoryBudgetError:
            render_memory_refusal(console)
            raise SystemExit(2) from None

        with console.status("Loading checkpoint onto the device…"):
            model, _ = load_llama_weights(snapshot, device=device, dtype=args.dtype)
        prompt = torch.tensor([prompt_ids], device=device, dtype=torch.long)
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold]generating"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            disable=args.no_progress,
        ) as progress:
            task = progress.add_task("generate", total=args.new_tokens)
            output = generate_cached(
                model,
                prompt,
                max_new_tokens=args.new_tokens,
                eos_token_id=checkpoint.eos_token_id,
                progress=lambda advance: progress.update(task, advance=advance),
            )
            generated_count = output.size(1) - len(prompt_ids)
            progress.update(task, total=generated_count, completed=generated_count)

        generated = output[0, len(prompt_ids) :].tolist()
        text = tokenizer.decode(generated)
        console.print(Panel(text or "[dim]<end of sequence>[/dim]", title="Completion"))
        if args.show_token_ids:
            console.print(f"[dim]token ids:[/dim] {generated}")
    except SystemExit:
        raise
    except KeyboardInterrupt:
        release_device_memory(device)
        console.print("\n[yellow]Generation cancelled; device cache released.[/yellow]")
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
