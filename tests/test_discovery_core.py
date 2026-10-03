from src.core.discovery import (
    AnomalySignal,
    DiscoveryCandidate,
    Observation,
)


def test_discovery_candidate_is_domain_agnostic():
    candidate = DiscoveryCandidate(
        observed_at="2026-01-01T12:00:00Z",
        domain="social",
        subject="example meme",
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
                entity="example meme",
                value=1200,
            )
        ],
        context={"topic_type": "unknown"},
    )

    data = candidate.to_dict()

    assert data["domain"] == "social"
    assert data["subject"] == "example meme"
    assert data["signals"][0]["name"] == "mention_velocity"
    assert data["observations"][0]["source"] == "reddit"


def test_market_is_just_another_domain():
    candidate = DiscoveryCandidate(
        observed_at="2026-01-01",
        domain="market",
        subject="HCLTECH",
        score=0.8,
        signals=[AnomalySignal(name="return_z", score=0.9)],
    )

    assert candidate.domain == "market"
    assert candidate.to_dict()["domain"] == "market"
