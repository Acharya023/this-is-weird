"""Smoke test for the domain-neutral discovery core."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.discovery import AnomalySignal, DiscoveryCandidate, Observation


def main():
    social = DiscoveryCandidate(
        observed_at="2026-01-01T12:00:00Z",
        domain="social",
        subject="example emerging topic",
        score=0.91,
        signals=[
            AnomalySignal(
                name="mention_velocity",
                score=0.88,
                direction="up",
                value=1200,
                baseline=100,
            )
        ],
        observations=[
            Observation(
                observed_at="2026-01-01T12:00:00Z",
                source="reddit",
                entity="example emerging topic",
                value=1200,
            )
        ],
    )

    market = DiscoveryCandidate(
        observed_at="2026-01-01",
        domain="market",
        subject="HCLTECH",
        score=0.80,
        signals=[AnomalySignal(name="return_z", score=0.90)],
    )

    assert social.to_dict()["domain"] == "social"
    assert social.to_dict()["signals"][0]["name"] == "mention_velocity"
    assert market.to_dict()["domain"] == "market"

    print("DOMAIN-NEUTRAL DISCOVERY CORE")
    print("SOCIAL:", social.to_dict())
    print("MARKET:", market.to_dict())
    print("PASS")


if __name__ == "__main__":
    main()
