"""Readable scheduling policies shared by the serving engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Schedulable(Protocol):
    arrival_step: int
    prompt_remaining: int
    last_token_step: int | None
    ttft_slo_steps: int | None
    itl_slo_steps: int | None


@dataclass(frozen=True)
class PrefillSlice:
    state: Schedulable
    tokens: int


@dataclass(frozen=True)
class Schedule:
    decode: tuple[Schedulable, ...]
    prefill: tuple[PrefillSlice, ...]

    @property
    def tokens(self) -> int:
        return len(self.decode) + sum(item.tokens for item in self.prefill)


class FCFSScheduler:
    """Decode-first FCFS with a token budget and bounded prefill chunks."""

    def __init__(
        self,
        *,
        max_batch_size: int = 32,
        max_batch_tokens: int = 512,
        prefill_chunk_size: int | None = None,
    ) -> None:
        if max_batch_size < 1 or max_batch_tokens < 1:
            raise ValueError("batch limits must be positive")
        if prefill_chunk_size is not None and prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be positive")
        self.max_batch_size = max_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.prefill_chunk_size = prefill_chunk_size

    def _decode_order(self, running: list[Schedulable], now: int) -> list[Schedulable]:
        del now
        return sorted(running, key=lambda state: state.arrival_step)

    def _prefill_order(
        self, prefilling: list[Schedulable], now: int
    ) -> list[Schedulable]:
        del now
        return sorted(prefilling, key=lambda state: state.arrival_step)

    def schedule(
        self,
        *,
        prefilling: list[Schedulable],
        running: list[Schedulable],
        now: int,
    ) -> Schedule:
        budget = self.max_batch_tokens
        slots = self.max_batch_size
        decode = self._decode_order(running, now)[: min(budget, slots)]
        budget -= len(decode)
        slots -= len(decode)

        prefill = []
        for state in self._prefill_order(prefilling, now):
            if budget == 0 or slots == 0:
                break
            chunk = state.prompt_remaining
            if self.prefill_chunk_size is not None:
                chunk = min(chunk, self.prefill_chunk_size)
            chunk = min(chunk, budget)
            if chunk:
                prefill.append(PrefillSlice(state, chunk))
                budget -= chunk
                slots -= 1
        return Schedule(tuple(decode), tuple(prefill))


class SLOAwareScheduler(FCFSScheduler):
    """Earliest-slack-first scheduling for TTFT and inter-token deadlines."""

    @staticmethod
    def _deadline(state: Schedulable, *, decode: bool) -> float:
        if decode:
            if state.itl_slo_steps is None:
                return float("inf")
            origin = state.last_token_step or state.arrival_step
            return origin + state.itl_slo_steps
        if state.ttft_slo_steps is None:
            return float("inf")
        return state.arrival_step + state.ttft_slo_steps

    def _decode_order(self, running: list[Schedulable], now: int) -> list[Schedulable]:
        return sorted(
            running,
            key=lambda state: (
                self._deadline(state, decode=True) - now,
                state.arrival_step,
            ),
        )

    def _prefill_order(
        self, prefilling: list[Schedulable], now: int
    ) -> list[Schedulable]:
        return sorted(
            prefilling,
            key=lambda state: (
                self._deadline(state, decode=False) - now,
                state.arrival_step,
            ),
        )
