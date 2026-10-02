"""Tests for the semantic investigation layer."""

import unittest

from src.investigation.semantic import SemanticEvidenceAnalyzer


class FakeEventClassifier:
    def __call__(self, text, labels, multi_label=False):
        return {
            "labels": [
                "government or regulatory policy",
                "earnings or quarterly results",
            ],
            "scores": [0.91, 0.04],
        }


class FakeSentimentClassifier:
    def __call__(self, text):
        return [{"label": "negative", "score": 0.88}]


class SemanticTests(unittest.TestCase):
    def test_semantic_event_classification_is_not_keyword_based(self):
        analyzer = SemanticEvidenceAnalyzer(
            event_classifier=FakeEventClassifier(),
            sentiment_classifier=FakeSentimentClassifier(),
        )

        result = analyzer.analyze({
            "title": "Auto stocks jump after major tax restructuring",
            "text": (
                "Maruti shares rose sharply as investors assessed the "
                "government's changes to the indirect-tax framework."
            ),
        })

        self.assertEqual(result.event_type, "policy_event")
        self.assertEqual(result.event_label, "government or regulatory policy")
        self.assertAlmostEqual(result.event_confidence, 0.91)
        self.assertEqual(result.sentiment, "negative")
        self.assertAlmostEqual(result.sentiment_confidence, 0.88)

    def test_empty_article_is_unknown(self):
        analyzer = SemanticEvidenceAnalyzer(
            event_classifier=FakeEventClassifier(),
            sentiment_classifier=FakeSentimentClassifier(),
        )
        result = analyzer.analyze({"title": "", "text": ""})
        self.assertEqual(result.event_type, "unknown")
        self.assertEqual(result.event_confidence, 0.0)


if __name__ == "__main__":
    unittest.main()


class FakeNERModel:
    def predict_entities(self, text, labels, threshold=0.5):
        return [
            {"text": "Bharti Airtel", "label": "company", "score": 0.94},
            {"text": "Singtel", "label": "organization", "score": 0.91},
            {"text": "5.1 crore", "label": "amount", "score": 0.89},
            {"text": "block trade", "label": "event_signal", "score": 0.86},
        ]


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=128):
        return {"input_ids": None}


class FakeEventModel:
    class Config:
        id2label = {
            0: "Deal",
            1: "Profit/Loss",
            2: "SecurityValue",
        }

    config = Config()

    def __call__(self, **inputs):
        class Output:
            # Sigmoid(3.0) ~= .953, sigmoid(.2) ~= .550, sigmoid(-2) ~= .119.
            logits = __import__("torch").tensor([[3.0, 0.2, -2.0]])
        return Output()


class FinancialNLPTests(unittest.TestCase):
    def test_financial_nlp_extracts_entities_and_multilabel_events(self):
        from src.investigation.semantic import FinancialNLPAnalyzer

        analyzer = FinancialNLPAnalyzer(
            ner_model=FakeNERModel(),
            event_model=FakeEventModel(),
            sentiment_classifier=FakeSentimentClassifier(),
        )
        analyzer._event_tokenizer = FakeTokenizer()

        result = analyzer.analyze({
            "title": "Bharti Airtel shares decline after large block deal",
            "text": "Singtel sold shares in a block trade.",
        })

        self.assertEqual(result.entities[0]["label"], "company")
        self.assertEqual(result.entities[1]["label"], "organization")
        self.assertEqual(result.event_labels[0]["label"], "Deal")
        self.assertEqual(result.event_labels[0]["mapped_event_type"], "corporate_action")
        self.assertEqual(result.event_labels[1]["label"], "Profit/Loss")
        self.assertEqual(result.event_labels[1]["mapped_event_type"], "earnings")
        self.assertEqual(result.sentiment, "negative")

    def test_financial_nlp_empty_article_does_not_load_models(self):
        from src.investigation.semantic import FinancialNLPAnalyzer

        analyzer = FinancialNLPAnalyzer()
        result = analyzer.analyze({"title": "", "text": ""})

        self.assertEqual(result.entities, [])
        self.assertEqual(result.event_labels, [])
        self.assertEqual(result.mapped_event_types, [])
        self.assertIsNone(result.sentiment)
