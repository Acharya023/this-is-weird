"""End-to-end smoke test for observation -> anomaly -> discovery."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.observations import build_observation_batch
from src.detection.generic_anomaly import AnomalyConfig
from src.detection.pipeline import detect_entity_discovery


def main():
    batch = build_observation_batch(
        [
            {"observed_at": "2026-01-01", "entity": "example", "value": 100},
            {"observed_at": "2026-01-02", "entity": "example", "value": 102},
            {"observed_at": "2026-01-03", "entity": "example", "value": 98},
            {"observed_at": "2026-01-04", "entity": "example", "value": 101},
            {"observed_at": "2026-01-05", "entity": "example", "value": 99},
            {"observed_at": "2026-01-06", "entity": "example", "value": 160},
        ],
        source="test-source",
        batch_metadata={"domain": "example"},
    )

    candidate = detect_entity_discovery(
        batch.observations,
        domain=batch.metadata["domain"],
        config=AnomalyConfig(min_history=5, z_threshold=3.0),
    )

    assert candidate is not None
    assert candidate.domain == "example"
    assert candidate.subject == "example"
    assert candidate.signals[0].name == "scalar_z_score"
    assert candidate.context["stage"] == "statistical_discovery"
    assert candidate.context["measurement"] is None
    assert candidate.score > 0

    print("DISCOVERY PIPELINE")
    print("CANDIDATE:", candidate.to_dict())
    print("PASS")


if __name__ == "__main__":
    main()
