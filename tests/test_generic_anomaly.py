"""Tests for the domain-neutral scalar anomaly detector."""

from src.core.discovery import Observation
from src.detection.generic_anomaly import AnomalyConfig, detect_scalar_anomaly


def _stream(values):
    return [
        Observation(
            observed_at=f"2026-01-{index + 1:02d}",
            source="test",
            entity="example",
            value=value,
        )
        for index, value in enumerate(values)
    ]


def test_detects_large_positive_deviation():
    signal = detect_scalar_anomaly(
        _stream([100, 102, 98, 101, 99, 160]),
        config=AnomalyConfig(min_history=5, z_threshold=3.0),
    )

    assert signal is not None
    assert signal.metadata["anomalous"] is True
    assert signal.direction == "up"
    assert signal.value == 160
    assert signal.baseline == 100


def test_ignores_normal_change():
    signal = detect_scalar_anomaly(
        _stream([100, 102, 98, 101, 99, 103]),
        config=AnomalyConfig(min_history=5, z_threshold=3.0),
    )

    assert signal is None


def test_requires_enough_history():
    signal = detect_scalar_anomaly(
        _stream([100, 102, 98, 160]),
        config=AnomalyConfig(min_history=5, z_threshold=3.0),
    )

    assert signal is None
