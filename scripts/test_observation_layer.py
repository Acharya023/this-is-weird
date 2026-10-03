"""Reproducible smoke test for the domain-neutral observation layer."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.observations import build_observation_batch


def main():
    market = build_observation_batch(
        [
            {
                "observed_at": "2026-01-02",
                "entity": "HCLTECH",
                "value": -0.084,
                "metadata": {"measurement": "daily_return"},
            },
            {
                "observed_at": "2026-01-02",
                "entity": "INFY",
                "value": -0.011,
                "metadata": {"measurement": "daily_return"},
            },
        ],
        source="nse",
        batch_metadata={"domain": "market"},
    )

    social = build_observation_batch(
        [
            {
                "observed_at": "2026-01-02T12:00:00Z",
                "entity": "example meme",
                "value": 1200,
                "metadata": {"measurement": "mention_count"},
            },
            {
                "observed_at": "2026-01-02T13:00:00Z",
                "entity": "example meme",
                "value": 2400,
                "metadata": {"measurement": "mention_count"},
            },
        ],
        source="reddit",
        batch_metadata={"domain": "social"},
    )

    assert market.source == "nse"
    assert market.metadata["domain"] == "market"
    assert market.observations[0].entity == "HCLTECH"
    assert market.observations[0].value == -0.084

    assert social.source == "reddit"
    assert social.metadata["domain"] == "social"
    assert social.observations[0].entity == "example meme"
    assert social.observations[1].value == 2400.0

    print("OBSERVATION LAYER")
    print("MARKET:", market)
    print("SOCIAL:", social)
    print("PASS")


if __name__ == "__main__":
    main()
