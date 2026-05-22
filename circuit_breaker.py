from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


@dataclass
class KillSwitchEvent:
    state: CircuitState
    reason: str
    bdi_score: float
    entropy: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "reason": self.reason,
            "bdi_score": self.bdi_score,
            "entropy": self.entropy,
            "timestamp": self.timestamp,
        }


StateListener = Callable[[KillSwitchEvent], None]


class CircuitBreakerMiddleware:
    def __init__(
        self,
        bdi_threshold: float = 0.65,
        entropy_threshold: float = 0.85,
        reset_timeout_seconds: int = 30,
    ):
        self.state: CircuitState = CircuitState.CLOSED
        self.bdi_threshold = bdi_threshold
        self.entropy_threshold = entropy_threshold
        self.reset_timeout_seconds = reset_timeout_seconds
        self.last_event: Optional[KillSwitchEvent] = None
        self.failure_count: int = 0
        self.last_failure: Optional[dict[str, object]] = None
        self._listeners: List[StateListener] = []

    def register_listener(self, listener: StateListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unregister_listener(self, listener: StateListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _emit(self, event: KillSwitchEvent) -> None:
        self.last_event = event
        for listener in list(self._listeners):
            listener(event)

    def open(self, reason: str, bdi_score: float, entropy: float) -> None:
        if self.state != CircuitState.OPEN:
            # Transitioning to OPEN because a kill condition was detected.
            self.state = CircuitState.OPEN
            self.failure_count += 1
            self.last_failure = {
                "reason": reason,
                "bdi_score": bdi_score,
                "entropy": entropy,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._emit(KillSwitchEvent(state=self.state, reason=reason, bdi_score=bdi_score, entropy=entropy))

    def close(self, reason: str, bdi_score: float, entropy: float) -> None:
        if self.state != CircuitState.CLOSED:
            # Transitioning to CLOSED once governance conditions are within safe bounds.
            self.state = CircuitState.CLOSED
            self._emit(KillSwitchEvent(state=self.state, reason=reason, bdi_score=bdi_score, entropy=entropy))

    def half_open(self, reason: str, bdi_score: float, entropy: float) -> None:
        if self.state != CircuitState.HALF_OPEN:
            # Transitioning to HALF_OPEN when a hold condition has been reached.
            self.state = CircuitState.HALF_OPEN
            self._emit(KillSwitchEvent(state=self.state, reason=reason, bdi_score=bdi_score, entropy=entropy))

    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def is_half_open(self) -> bool:
        return self.state == CircuitState.HALF_OPEN

    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    def evaluate(self, bdi_score: float, entropy: float) -> Dict[str, object]:
        if entropy > self.entropy_threshold:
            self.open(
                reason="Shannon entropy exceeded kill threshold",
                bdi_score=bdi_score,
                entropy=entropy,
            )
            return {
                "action": "kill",
                "reason": "Governance kill switch engaged due to high entropy.",
                "state": self.state.value,
            }

        if bdi_score > self.bdi_threshold:
            self.half_open(
                reason="BDI exceeded hold threshold",
                bdi_score=bdi_score,
                entropy=entropy,
            )
            return {
                "action": "hold",
                "reason": "Governance hold engaged due to elevated BDI.",
                "state": self.state.value,
            }

        self.close(
            reason="Governance conditions returned to safe bounds",
            bdi_score=bdi_score,
            entropy=entropy,
        )
        return {
            "action": "proceed",
            "reason": "Governance conditions are within safe bounds.",
            "state": self.state.value,
        }

    def status(self) -> Dict[str, object]:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure,
        }
