"""Domain-neutral anomaly detection for scalar observation streams."""

from dataclasses import dataclass
from math import isfinite, sqrt
from statistics import mean, pstdev
from typing import Iterable

from src.core.discovery import AnomalySignal, Observation


@dataclass(frozen=True)
class AnomalyConfig:
    """Controls the statistical baseline used by the generic detector."""

    min_history: int = 5
    z_threshold: float = 3.0
    epsilon: float = 1e-12


def _z_score(value: float, history: list[float], epsilon: float) -> float:
    baseline_mean = mean(history)
    deviation = pstdev(history)
    return (value - baseline_mean) / max(deviation, epsilon)


def detect_scalar_anomaly(
    observations: Iterable[Observation],
    *,
    config: AnomalyConfig | None = None,
) -> AnomalySignal | None:
    """Detect whether the latest scalar observation is unusually large/small.

    Observations are assumed to represent one comparable measurement stream.
    The latest observation is never included in its own baseline.
    """
    config = config or AnomalyConfig()
    ordered = list(observations)

    if len(ordered) < config.min_history + 1:
        return None

    latest = ordered[-1]
    if latest.value is None or not isfinite(latest.value):
        return None

    history = [
        observation.value
        for observation in ordered[:-1]
        if observation.value is not None and isfinite(observation.value)
    ]

    if len(history) < config.min_history:
        return None

    baseline = mean(history)
    z_score = _z_score(latest.value, history, config.epsilon)
    magnitude = abs(z_score)

    if magnitude < config.z_threshold:
        return None

    # Convert z-score magnitude into a bounded anomaly score.
    score = min(magnitude / (config.z_threshold * 2.0), 1.0)

    return AnomalySignal(
        name="scalar_z_score",
        score=score,
        direction="up" if latest.value > baseline else "down",
        value=latest.value,
        baseline=baseline,
        metadata={
            "z_score": z_score,
            "history_size": len(history),
            "anomalous": True,
        },
    )
