"""Domain-agnostic discovery primitives for This Is Weird.

A discovery is an unusual change in an observed stream. The stream may be
financial, social, news, entertainment, sports, technology, or another domain.
Domain-specific detectors produce anomaly signals; investigation determines
what the anomaly means.
"""

from dataclasses import dataclass, field
from typing import Any

from .observations import Observation


@dataclass
class AnomalySignal:
    """A signal that an observation changed unusually."""

    name: str
    score: float
    direction: str | None = None
    value: float | None = None
    baseline: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryCandidate:
    """A domain-neutral candidate passed to investigation."""

    observed_at: str
    domain: str
    subject: str
    score: float
    signals: list[AnomalySignal] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)

    # Context that helps decide whether the anomaly deserves human attention.
    # These are descriptive inputs, not an overall human-facing verdict.
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_at": self.observed_at,
            "domain": self.domain,
            "subject": self.subject,
            "score": self.score,
            "signals": [
                {
                    "name": signal.name,
                    "score": signal.score,
                    "direction": signal.direction,
                    "value": signal.value,
                    "baseline": signal.baseline,
                    "metadata": signal.metadata,
                }
                for signal in self.signals
            ],
            "observations": [
                {
                    "observed_at": observation.observed_at,
                    "source": observation.source,
                    "entity": observation.entity,
                    "value": observation.value,
                    "metadata": observation.metadata,
                }
                for observation in self.observations
            ],
            "context": self.context,
        }
