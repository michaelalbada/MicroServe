"""Stage 8 routing: predict latency instead of counting queued requests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteRequest:
    ready_at_ms: float
    tokens: int
    deadline_ms: float | None = None


@dataclass(frozen=True)
class ReplicaState:
    name: str
    available_at_ms: float
    tokens_per_ms: float
    fixed_ms: float = 0.0
    cached_prefix_tokens: int = 0
    transfer_ms: float = 0.0

    def predict(self, request: RouteRequest) -> tuple[float, float]:
        remaining = max(0, request.tokens - self.cached_prefix_tokens)
        start = max(request.ready_at_ms, self.available_at_ms)
        finish = start + self.fixed_ms + remaining / self.tokens_per_ms
        finish += self.transfer_ms
        return start, finish


@dataclass(frozen=True)
class RouteDecision:
    replica: ReplicaState
    predicted_start_ms: float
    predicted_finish_ms: float
    predicted_deadline_miss_ms: float


class QueueOnlyRouter:
    """A baseline that sees worker availability but not service cost/locality."""

    def route(
        self, request: RouteRequest, replicas: list[ReplicaState]
    ) -> RouteDecision:
        if not replicas:
            raise ValueError("at least one replica is required")
        replica = min(replicas, key=lambda item: (item.available_at_ms, item.name))
        start, finish = replica.predict(request)
        miss = (
            max(0.0, finish - request.deadline_ms)
            if request.deadline_ms is not None
            else 0.0
        )
        return RouteDecision(replica, start, finish, miss)


class SLOAwareRouter:
    """Route by predicted completion, including cache and transfer effects."""

    def route(
        self, request: RouteRequest, replicas: list[ReplicaState]
    ) -> RouteDecision:
        if not replicas:
            raise ValueError("at least one replica is required")

        decisions = []
        for replica in replicas:
            start, finish = replica.predict(request)
            miss = (
                max(0.0, finish - request.deadline_ms)
                if request.deadline_ms is not None
                else 0.0
            )
            decisions.append(RouteDecision(replica, start, finish, miss))
        return min(
            decisions,
            key=lambda decision: (
                decision.predicted_deadline_miss_ms > 0,
                decision.predicted_deadline_miss_ms,
                decision.predicted_finish_ms,
                decision.replica.name,
            ),
        )
