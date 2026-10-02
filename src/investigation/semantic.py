"""Semantic analysis for investigation evidence.

This module deliberately keeps model inference separate from deterministic
market/evidence scoring. It uses Hugging Face models to interpret article
meaning rather than relying on keyword matches.
"""

from dataclasses import dataclass
from typing import Any


EVENT_LABELS = [
    "earnings or quarterly results",
    "management guidance or forecast",
    "corporate action",
    "block or bulk trade",
    "government or regulatory policy",
    "business or operational update",
    "analyst or brokerage research",
    "sector or macroeconomic event",
    "other event",
]

EVENT_TYPE_MAP = {
    "earnings or quarterly results": "earnings",
    "management guidance or forecast": "guidance",
    "corporate action": "corporate_action",
    "block or bulk trade": "block_trade",
    "government or regulatory policy": "policy_event",
    "business or operational update": "business_update",
    "analyst or brokerage research": "analyst_research",
    "sector or macroeconomic event": "macro_or_sector_event",
    "other event": "unknown",
}


@dataclass
class SemanticAnalysis:
    event_type: str
    event_label: str
    event_confidence: float
    sentiment: str | None = None
    sentiment_confidence: float | None = None


class SemanticEvidenceAnalyzer:
    """Interpret financial evidence with Hugging Face NLP models.

    Models are loaded lazily so importing the repository does not download
    large model weights. The classifier can also be injected for tests.
    """

    def __init__(
        self,
        event_classifier: Any | None = None,
        sentiment_classifier: Any | None = None,
        event_model: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        sentiment_model: str = "ProsusAI/finbert",
    ):
        self._event_classifier = event_classifier
        self._sentiment_classifier = sentiment_classifier
        self.event_model = event_model
        self.sentiment_model = sentiment_model

    def _load_event_classifier(self):
        if self._event_classifier is None:
            from transformers import pipeline

            self._event_classifier = pipeline(
                "zero-shot-classification",
                model=self.event_model,
            )
        return self._event_classifier

    def _load_sentiment_classifier(self):
        if self._sentiment_classifier is None:
            from transformers import pipeline

            self._sentiment_classifier = pipeline(
                "text-classification",
                model=self.sentiment_model,
            )
        return self._sentiment_classifier

    @staticmethod
    def _article_text(article: dict) -> str:
        title = article.get("title", "")
        text = article.get("text", "")
        # Keep inference bounded while preserving the headline and opening
        # portion, where news articles usually state the main event.
        combined = f"{title}. {text}".strip()
        return combined[:6000]

    def analyze(self, article: dict) -> SemanticAnalysis:
        text = self._article_text(article)
        if not text:
            return SemanticAnalysis("unknown", "other event", 0.0)

        event_result = self._load_event_classifier()(
            text,
            EVENT_LABELS,
            multi_label=False,
        )
        label = event_result["labels"][0]
        confidence = float(event_result["scores"][0])
        event_type = EVENT_TYPE_MAP[label]

        sentiment_result = self._load_sentiment_classifier()(text)[0]
        sentiment = str(sentiment_result["label"]).lower()
        sentiment_confidence = float(sentiment_result["score"])

        return SemanticAnalysis(
            event_type=event_type,
            event_label=label,
            event_confidence=confidence,
            sentiment=sentiment,
            sentiment_confidence=sentiment_confidence,
        )
