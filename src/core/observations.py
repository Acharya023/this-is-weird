"""Domain-neutral observation ingestion primitives.

Source adapters normalize raw records into Observations. The core does not
assume what an observation means; anomaly detectors interpret the measurements.
"""

from dataclasses import dataclass, field
from typing import Any, Iterable

from .discovery import Observation


@dataclass(frozen=True)
class ObservationBatch:
    """A validated collection of observations from one source adapter."""

    source: str
    observations: tuple[Observation, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_observation(
    *,
    observed_at: str,
    source: str,
    entity: str | None = None,
    value: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> Observation:
    """Convert one raw measurement into the common observation contract."""
    if not observed_at:
        raise ValueError("observed_at is required")
    if not source:
        raise ValueError("source is required")

    return Observation(
        observed_at=str(observed_at),
        source=str(source),
        entity=str(entity) if entity is not None else None,
        value=float(value) if value is not None else None,
        metadata=dict(metadata or {}),
    )


def build_observation_batch(
    records: Iterable[dict[str, Any]],
    *,
    source: str,
    batch_metadata: dict[str, Any] | None = None,
) -> ObservationBatch:
    """Normalize raw source records into one observation batch."""
    observations = tuple(
        normalize_observation(
            observed_at=record.get("observed_at"),
            source=source,
            entity=record.get("entity"),
            value=record.get("value"),
            metadata=record.get("metadata"),
        )
        for record in records
    )

    return ObservationBatch(
        source=source,
        observations=observations,
        metadata=dict(batch_metadata or {}),
    )
