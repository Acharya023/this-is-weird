"""Semantic financial NLP for investigation evidence.

The deterministic investigation layer finds unusual market behaviour.
This module interprets financial text without making model output the final
truth. The NLP layer combines a small Indian-financial company NER model with
deterministic extraction of financial facts, a financial event classifier, and
FinBERT sentiment.

Models:
- Indian financial company NER: small BERT token classifier.
- FinDeBERTa: multi-label financial event detection.
- FinBERT: financial sentiment baseline.

Deterministic verification remains outside this module.
"""

from dataclasses import dataclass, field
import re
from typing import Any


NER_MODEL = "ritam-m/bert-base-company-ner"
EVENT_MODEL = "ritessshhh/FinDeBERTa"
SENTIMENT_MODEL = "ProsusAI/finbert"

# Kept as a compatibility alias for callers that used the old constant.
GLINER_MODEL = NER_MODEL

FINANCIAL_FACT_PATTERNS = {
    "percentage": [
        r"(?<!\\w)(?:\\d+(?:\\.\\d+)?|\\.\\d+)\\s*%",
        r"(?<!\\w)(?:\\d+(?:\\.\\d+)?)\\s*percent\\b",
    ],
    "amount": [
        r"(?<!\\w)₹\\s*[\\d,.]+(?:\\s*(?:crore|cr|lakh|million|billion))?",
        r"(?<!\\w)\\$\\s*[\\d,.]+(?:\\s*(?:million|billion))?",
        r"(?<!\\w)[\\d,.]+\\s*(?:crore|cr|lakh|million|billion)\\b",
    ],
    "date": [
        r"\\b(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|Sept|September|Oct|October|Nov|November|Dec|December)\\s+\\d{1,2}(?:,\\s*\\d{4})?\\b",
        r"\\b\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}\\b",
    ],
}

EVENT_SIGNAL_PATTERNS = [
    "block deal",
    "bulk deal",
    "stake sale",
    "stake sold",
    "sold stake",
    "guidance",
    "outlook",
    "forecast",
    "raised guidance",
    "cut guidance",
    "lowered guidance",
    "earnings",
    "quarterly results",
    "q1 results",
    "q2 results",
    "q3 results",
    "q4 results",
    "business update",
    "sales update",
    "demerger",
    "rights issue",
    "buyback",
    "bonus issue",
    "stock split",
    "dividend",
    "gst reform",
    "tax reform",
    "policy announcement",
    "regulatory change",
]

FINANCIAL_TERM_PATTERNS = [
    "revenue",
    "profit",
    "net profit",
    "ebitda",
    "margin",
    "aum",
    "nim",
    "nii",
    "bad loans",
    "credit costs",
    "asset quality",
    "target price",
    "price target",
    "stake",
    "shares",
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
    entities: list[dict] = field(default_factory=list)
    event_labels: list[dict] = field(default_factory=list)
    mapped_event_types: list[str] = field(default_factory=list)
    financial_facts: list[dict] = field(default_factory=list)
    sentiment: str | None = None
    sentiment_confidence: float | None = None


class FinancialNLPAnalyzer:
    """Run financial entity/fact extraction, events and sentiment."""

    def __init__(
        self,
        ner_model: Any | None = None,
        event_model: Any | None = None,
        sentiment_classifier: Any | None = None,
        ner_model_name: str = NER_MODEL,
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
            from transformers import pipeline

            self._ner_model = pipeline(
                "ner",
                model=self.ner_model_name,
                aggregation_strategy="first",
            )
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

    @staticmethod
    def _regex_facts(text: str) -> list[dict]:
        facts = []
        seen = set()

        for label, patterns in FINANCIAL_FACT_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    value = match.group(0).strip()
                    key = (label, value.lower())
                    if key not in seen:
                        facts.append({"label": label, "text": value, "score": 1.0})
                        seen.add(key)

        lowered = text.lower()
        for phrase in EVENT_SIGNAL_PATTERNS:
            if phrase in lowered:
                key = ("event_signal", phrase)
                if key not in seen:
                    facts.append({"label": "event_signal", "text": phrase, "score": 1.0})
                    seen.add(key)

        for phrase in FINANCIAL_TERM_PATTERNS:
            if re.search(r"(?<!\\w)" + re.escape(phrase) + r"(?!\\w)", lowered):
                key = ("financial_term", phrase)
                if key not in seen:
                    facts.append({"label": "financial_term", "text": phrase, "score": 1.0})
                    seen.add(key)

        return facts

    def extract_entities(self, text: str) -> list[dict]:
        if not text:
            return []

        model = self._load_ner_model()

        # Keep the old fake-model contract used by unit tests.
        if hasattr(model, "predict_entities"):
            return model.predict_entities(
                text,
                ["company"],
                threshold=0.50,
            )

        raw = model(text)
        entities = []
        for item in raw:
            entities.append(
                {
                    "text": item.get("word", "").strip(),
                    "label": str(item.get("entity_group", "entity")).lower(),
                    "score": float(item.get("score", 0.0)),
                }
            )
        return entities

    def extract_financial_facts(self, text: str) -> list[dict]:
        if not text:
            return []
        return self._regex_facts(text)

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
        facts = self.extract_financial_facts(text)
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
            financial_facts=facts,
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
