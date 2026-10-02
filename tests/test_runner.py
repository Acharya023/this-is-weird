"""Tests for the automatic investigation runner."""

import unittest
from datetime import date

from src.investigation.runner import investigate_with_provider


class FakeProvider:
    def search(self, query, *, limit=5):
        return [{
            "title": "Kotak Bank earnings spark concerns",
            "text": "Kotak Bank shares fall after weak earnings.",
            "published_date": "2025-07-28",
            "url": "https://example.com/kotak",
            "source_name": "Moneycontrol",
        }]


class RunnerTests(unittest.TestCase):
    def test_runner_uses_provider_and_event_pipeline(self):
        row = {
            "symbol": "KOTAKBANK",
            "date": date(2025, 7, 28),
            "daily_return": -0.074367,
            "return_z": 5.029075,
            "volume_ratio": 5.904156,
            "market_divergence": -0.068082,
            "discovery_score": 0.830647,
        }
        result = investigate_with_provider(row, FakeProvider(), limit_per_query=5)
        self.assertEqual(result["symbol"], "KOTAKBANK")
        self.assertEqual(result["search_candidates"], 2)
        self.assertEqual(result["best_event_type"], "earnings")
        self.assertIn("event_groups", result)


if __name__ == "__main__":
    unittest.main()
