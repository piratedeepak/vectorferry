"""Progress events. The engine emits them; the CLI draws progress bars, the UI streams them."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Event:
    kind: str  # collection_started | batch_done | retry | collection_done | warning | error
    collection: str
    done: int = 0
    total: int | None = None
    message: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


Emit = Callable[[Event], None]


def noop(_: Event) -> None:
    pass
