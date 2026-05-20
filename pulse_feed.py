from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List


@dataclass
class PulseEvent:
    kind: str
    text: str
    data: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PulseFeed:
    def __init__(self, max_events: int = 500):
        self._events: Deque[PulseEvent] = deque(maxlen=max_events)

    @property
    def count(self) -> int:
        return len(self._events)

    def add(self, event: PulseEvent) -> None:
        self._events.append(event)

    def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(limit, 500))
        return [asdict(event) for event in list(self._events)[-safe_limit:]][::-1]

    def recent_texts(self, limit: int = 25) -> List[str]:
        safe_limit = max(1, min(limit, 500))
        return [event.text for event in list(self._events)[-safe_limit:] if event.text]

    def clear(self) -> None:
        self._events.clear()
