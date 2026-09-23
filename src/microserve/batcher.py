"""A continuous-batching engine with explicit request lifecycle state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import torch

from microserve.kv_cache import Cache
from microserve.model import Transformer
from microserve.scheduler import FCFSScheduler, Schedule


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt: torch.Tensor
    max_new_tokens: int
    arrival_step: int = 0
    ttft_slo_steps: int | None = None
    itl_slo_steps: int | None = None

    def __post_init__(self) -> None:
        if self.prompt.ndim != 1 or self.prompt.numel() == 0:
            raise ValueError("prompt must be a non-empty one-dimensional tensor")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.arrival_step < 0:
            raise ValueError("arrival_step must be non-negative")


@dataclass
class RequestState:
    request: Request
    cache: Cache
    prompt_offset: int = 0
    generated: list[int] = field(default_factory=list)
    token_steps: list[int] = field(default_factory=list)

    @property
    def arrival_step(self) -> int:
        return self.request.arrival_step

    @property
    def prompt_remaining(self) -> int:
        return self.request.prompt.numel() - self.prompt_offset

    @property
    def last_token_step(self) -> int | None:
        return self.token_steps[-1] if self.token_steps else None

    @property
    def ttft_slo_steps(self) -> int | None:
        return self.request.ttft_slo_steps

    @property
    def itl_slo_steps(self) -> int | None:
        return self.request.itl_slo_steps


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    prompt: tuple[int, ...]
    generated: tuple[int, ...]
    arrival_step: int
    token_steps: tuple[int, ...]

    @property
    def first_token_step(self) -> int:
        return self.token_steps[0]

    @property
    def completion_step(self) -> int:
        return self.token_steps[-1]

    @property
    def ttft_steps(self) -> int:
        return self.first_token_step - self.arrival_step

    @property
    def itl_steps(self) -> tuple[int, ...]:
        return tuple(
            right - left for left, right in zip(self.token_steps, self.token_steps[1:])
        )


@dataclass(frozen=True)
class StepResult:
    step: int
    scheduled: Schedule
    completed: tuple[str, ...]


CacheFactory = Callable[[Request], Cache]
TokenCallback = Callable[[Request, int], None]


class PrefixStore(Protocol):
    def lookup(
        self, prompt: torch.Tensor, *, max_tokens: int
    ) -> tuple[Cache, int] | None: ...

    def insert(self, prompt: torch.Tensor, cache: Cache) -> None: ...


class ContinuousBatcher:
    """Admit, prefill, decode, and retire requests on every engine iteration."""

    def __init__(
        self,
        model: Transformer,
        *,
        scheduler: FCFSScheduler | None = None,
        cache_factory: CacheFactory | None = None,
        prefix_cache: PrefixStore | None = None,
        token_callback: TokenCallback | None = None,
    ) -> None:
        self.model = model
        self.scheduler = scheduler or FCFSScheduler()
        self.cache_factory = cache_factory or self._contiguous_cache
        self.prefix_cache = prefix_cache
        self.token_callback = token_callback
        self.clock = 0
        self.pending: list[Request] = []
        self.prefilling: list[RequestState] = []
        self.running: list[RequestState] = []
        self.results: dict[str, RequestResult] = {}
        self._request_ids: set[str] = set()

    def _contiguous_cache(self, request: Request) -> Cache:
        return self.model.new_cache(
            batch_size=1,
            max_tokens=request.prompt.numel() + request.max_new_tokens,
        )

    def submit(self, request: Request) -> None:
        if request.request_id in self._request_ids:
            raise ValueError(f"duplicate request id: {request.request_id}")
        if (
            request.prompt.numel() + request.max_new_tokens
            > self.model.config.max_seq_len
        ):
            raise ValueError("request exceeds the model sequence limit")
        self._request_ids.add(request.request_id)
        if request.arrival_step <= self.clock:
            self._admit(request)
        else:
            self.pending.append(request)

    def _admit(self, request: Request) -> None:
        max_tokens = request.prompt.numel() + request.max_new_tokens
        cache = self.cache_factory(request)
        prompt_offset = 0
        if self.prefix_cache is not None:
            match = self.prefix_cache.lookup(request.prompt, max_tokens=max_tokens)
            if match is not None:
                cache.close()
                cache, prompt_offset = match
        self.prefilling.append(
            RequestState(request, cache, prompt_offset=prompt_offset)
        )

    @staticmethod
    def _sample(logits: torch.Tensor) -> int:
        return int(logits[0, -1].argmax().item())

    def _emit(self, state: RequestState, token: int, at_step: int) -> bool:
        state.generated.append(token)
        state.token_steps.append(at_step)
        if self.token_callback is not None:
            self.token_callback(state.request, token)
        return len(state.generated) == state.request.max_new_tokens

    def _finish(self, state: RequestState) -> None:
        state.cache.close()
        request = state.request
        self.results[request.request_id] = RequestResult(
            request_id=request.request_id,
            prompt=tuple(int(token) for token in request.prompt.tolist()),
            generated=tuple(state.generated),
            arrival_step=request.arrival_step,
            token_steps=tuple(state.token_steps),
        )

    @torch.inference_mode()
    def step(self) -> StepResult:
        arriving = [
            request for request in self.pending if request.arrival_step <= self.clock
        ]
        for request in arriving:
            self.pending.remove(request)
            self._admit(request)
        schedule = self.scheduler.schedule(
            prefilling=self.prefilling,
            running=self.running,
            now=self.clock,
        )
        completed: list[str] = []
        emitted_at = self.clock + 1

        decode_states = list(schedule.decode)
        if decode_states:
            tokens = torch.tensor(
                [[state.generated[-1]] for state in decode_states],
                device=decode_states[0].request.prompt.device,
                dtype=torch.long,
            )
            logits = self.model.decode_batch(
                tokens, caches=[state.cache for state in decode_states]
            )
            for row, state in enumerate(decode_states):
                token = int(logits[row, -1].argmax().item())
                if self._emit(state, token, emitted_at):
                    self.running.remove(state)
                    self._finish(state)
                    completed.append(state.request.request_id)

        for item in schedule.prefill:
            state = item.state
            start = state.prompt_offset
            end = start + item.tokens
            chunk = state.request.prompt[start:end][None, :]
            logits = self.model(chunk, cache=state.cache, last_token_only=True)
            state.prompt_offset = end
            if state.prompt_remaining == 0:
                self.prefilling.remove(state)
                if self.prefix_cache is not None:
                    self.prefix_cache.insert(state.request.prompt, state.cache)
                token = self._sample(logits)
                if self._emit(state, token, emitted_at):
                    self._finish(state)
                    completed.append(state.request.request_id)
                else:
                    self.running.append(state)

        result = StepResult(self.clock, schedule, tuple(completed))
        self.clock += 1
        return result

    def run(self, requests: list[Request]) -> dict[str, RequestResult]:
        for request in requests:
            self.submit(request)
        while len(self.results) < len(requests):
            self.step()
        return self.results
