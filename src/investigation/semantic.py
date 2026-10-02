"""Semantic financial NLP for investigation evidence.

The deterministic investigation layer finds unusual market behaviour.
This module provides a separate NLP layer that interprets financial text.

Models:
- Indian-financial GLiNER: entity/event-signal extraction.
- FinDeBERTa: multi-label financial event detection.
- FinBERT: financial sentiment baseline.

No model output is treated as ground truth. Deterministic verification remains
outside this module.
"""

from dataclasses import dataclass, field
from typing import Any


GLINER_MODEL = "techkiyan/indian-financial-news-ner-gliner-v1"
EVENT_MODEL = "ritessshhh/FinDeBERTa"
SENTIMENT_MODEL = "ProsusAI/finbert"

GLINER_LABELS = [
    "company",
    "index",
    "organization",
    "person",
    "amount",
    "percentage",
    "date",
    "financial_term",
    "event_signal",
    "sector",
]

EVENT_TYPE_MAP = {
    "FinancialReport": "earnings",
    "Profit/Loss": "earnings",
    "Revenue": "earnings",
    "Dividend": "corporate_action",
    "Merger/Acquisition": "corporate_action",
    "Deal": "corporate_action",
    "Financing": "corporate_action",
    "Rating": "analyst_research",
    "Macroeconomics": "macro_or_sector_event",
    "SalesVolume": "business_update",
    "Product/Service": "business_update",
    "Employment": "business_update",
    "Expense": "unknown",
    "SecurityValue": "unknown",
    "CSR/Brand": "business_update",
}

# Kept for backwards compatibility with the earlier zero-shot experiment.
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

EVENT_TYPE_MAP_LEGACY = {
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


@dataclass
class FinancialNLPAnalysis:
    """Structured output from the new financial NLP layer."""

    entities: list[dict] = field(default_factory=list)
    event_labels: list[dict] = field(default_factory=list)
    mapped_event_types: list[str] = field(default_factory=list)
    sentiment: str | None = None
    sentiment_confidence: float | None = None


class FinancialNLPAnalyzer:
    """Run the financial NER, event classifier and optional FinBERT baseline."""

    def __init__(
        self,
        ner_model: Any | None = None,
        event_model: Any | None = None,
        sentiment_classifier: Any | None = None,
        ner_model_name: str = GLINER_MODEL,
        event_model_name: str = EVENT_MODEL,
        sentiment_model_name: str = SENTIMENT_MODEL,
        enable_sentiment: bool = True,
    ):
        self._ner_model = ner_model
        self._event_model = event_model
        self._sentiment_classifier = sentiment_classifier
        self.ner_model_name = ner_model_name
        self.event_model_name = event_model_name
        self.sentiment_model_name = sentiment_model_name
        self.enable_sentiment = enable_sentiment
        self._event_tokenizer = None

    @staticmethod
    def article_text(article: dict) -> str:
        title = str(article.get("title") or "").strip()
        text = str(article.get("text") or "").strip()
        return ". ".join(part for part in (title, text) if part)[:6000]

    def _load_ner_model(self):
        if self._ner_model is None:
            from gliner import GLiNER

            self._ner_model = GLiNER.from_pretrained(self.ner_model_name)
        return self._ner_model

    def _load_event_model(self):
        if self._event_model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._event_tokenizer = AutoTokenizer.from_pretrained(self.event_model_name)
            self._event_model = AutoModelForSequenceClassification.from_pretrained(
                self.event_model_name
            )
        return self._event_model

    def _load_sentiment(self):
        if self._sentiment_classifier is None:
            from transformers import pipeline

            self._sentiment_classifier = pipeline(
                "text-classification",
                model=self.sentiment_model_name,
            )
        return self._sentiment_classifier

    def extract_entities(self, text: str) -> list[dict]:
        if not text:
            return []
        model = self._load_ner_model()
        return model.predict_entities(text, GLINER_LABELS, threshold=0.50)

    def detect_events(self, text: str) -> list[dict]:
        if not text:
            return []

        import torch

        model = self._load_event_model()
        tokenizer = self._event_tokenizer
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            probabilities = torch.sigmoid(model(**inputs).logits)[0].cpu().tolist()

        labels = model.config.id2label
        predictions = [
            {
                "label": labels[i],
                "score": float(probabilities[i]),
                "mapped_event_type": EVENT_TYPE_MAP.get(labels[i], "unknown"),
            }
            for i in range(len(probabilities))
            if probabilities[i] >= 0.50
        ]
        return sorted(predictions, key=lambda item: item["score"], reverse=True)

    def analyze(self, article: dict) -> FinancialNLPAnalysis:
        text = self.article_text(article)
        if not text:
            return FinancialNLPAnalysis()

        entities = self.extract_entities(text)
        event_labels = self.detect_events(text)

        sentiment = None
        sentiment_confidence = None
        if self.enable_sentiment:
            result = self._load_sentiment()(text)[0]
            sentiment = str(result["label"]).lower()
            sentiment_confidence = float(result["score"])

        mapped = []
        for item in event_labels:
            event_type = item["mapped_event_type"]
            if event_type != "unknown" and event_type not in mapped:
                mapped.append(event_type)

        return FinancialNLPAnalysis(
            entities=entities,
            event_labels=event_labels,
            mapped_event_types=mapped,
            sentiment=sentiment,
            sentiment_confidence=sentiment_confidence,
        )


class SemanticEvidenceAnalyzer:
    """Legacy zero-shot + FinBERT analyzer retained for comparison tests."""

    def __init__(
        self,
        event_classifier: Any | None = None,
        sentiment_classifier: Any | None = None,
        event_model: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        sentiment_model: str = SENTIMENT_MODEL,
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
        title = str(article.get("title") or "").strip()
        text = str(article.get("text") or "").strip()
        return ". ".join(part for part in (title, text) if part)[:6000]

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
        event_type = EVENT_TYPE_MAP_LEGACY[label]

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
