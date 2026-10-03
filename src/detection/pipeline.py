"""Turn anomaly signals into domain-neutral discovery candidates."""

from typing import Iterable

from src.core.discovery import DiscoveryCandidate, Observation
from src.detection.generic_anomaly import AnomalyConfig, detect_scalar_anomaly


def detect_entity_discovery(
    observations: Iterable[Observation],
    *,
    domain: str,
    config: AnomalyConfig | None = None,
) -> DiscoveryCandidate | None:
    """Detect an anomaly in one entity stream and package it for investigation."""
    ordered = list(observations)
    if not ordered:
        return None

    signal = detect_scalar_anomaly(ordered, config=config)
    if signal is None:
        return None

    latest = ordered[-1]

    return DiscoveryCandidate(
        observed_at=latest.observed_at,
        domain=domain,
        subject=latest.entity or "unknown",
        score=signal.score,
        signals=[signal],
        observations=ordered,
        context={
            "source": latest.source,
            "measurement": latest.metadata.get("measurement"),
            "history_size": signal.metadata.get("history_size"),
            "z_score": signal.metadata.get("z_score"),
            "stage": "statistical_discovery",
        },
    )
