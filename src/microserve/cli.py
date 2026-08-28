"""The unified ``microserve`` command."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version

from microserve import benchmark, checkpoint, infer, quickstart, scorecard


Command = Callable[..., None]


@dataclass(frozen=True)
class CommandSpec:
    help: str
    main: Command


COMMANDS = {
    "quickstart": CommandSpec("Run a fast KV-cache crossover demo.", quickstart.main),
    "generate": CommandSpec("Generate text with a real model.", infer.main),
    "bench": CommandSpec("Benchmark naive and cached generation.", benchmark.main),
    "scorecard": CommandSpec("Run the complete serving curriculum.", scorecard.main),
    "fetch": CommandSpec(
        "Fetch a supported model into the local cache.", checkpoint.main
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microserve",
        description="Modern LLM serving from first principles.",
    )
    parser.add_argument(
        "--version", action="version", version=version("microserve-llm")
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, spec in COMMANDS.items():
        subparsers.add_parser(name, add_help=False, help=spec.help)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args, remaining = parser.parse_known_args(argv)
    if args.command is None:
        parser.print_help()
        return
    command = COMMANDS[args.command]
    command.main(remaining, prog=f"microserve {args.command}")


if __name__ == "__main__":
    main()
