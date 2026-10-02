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
