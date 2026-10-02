"""Tests for the investigation evidence layer."""

import unittest
from datetime import date

from src.investigation.evidence import (
    build_evidence_candidate,
    calculate_evidence_strength,
    consolidate_evidence,
)


class EvidenceTests(unittest.TestCase):

    def test_same_day_matching_event_scores_strongly(self):
        candidate = build_evidence_candidate(
            symbol="MARUTI",
            anomaly_date=date(2025, 8, 18),
            stock_return=0.0875,
            event_date=date(2025, 8, 18),
            event_type="policy_event",
            headline="Maruti shares rally on GST cuts",
            summary="Auto stocks rose sharply.",
            source_name="Reuters",
            source_url="https://www.reuters.com/example",
        )

        self.assertTrue(candidate["direction_matches"])
        self.assertGreater(candidate["evidence_relevance_score"], 0.85)

    def test_reuters_syndication_is_one_source_family(self):
        candidates = [
            build_evidence_candidate(
                "MARUTI",
                date(2025, 8, 18),
                0.08,
                date(2025, 8, 18),
                "policy_event",
                "Maruti rallies",
                "GST reform boosts autos",
                "Reuters",
                "https://www.reuters.com/example",
            ),
            build_evidence_candidate(
                "MARUTI",
                date(2025, 8, 18),
                0.08,
                date(2025, 8, 18),
                "policy_event",
                "Maruti rallies",
                "GST reform boosts autos",
                "Yahoo Finance",
                "https://finance.yahoo.com/example",
            ),
            build_evidence_candidate(
                "MARUTI",
                date(2025, 8, 18),
                0.08,
                date(2025, 8, 18),
                "policy_event",
                "Auto stocks rally",
                "GST reform boosts autos",
                "Moneycontrol",
                "https://www.moneycontrol.com/example",
            ),
        ]

        consolidated = consolidate_evidence(candidates)

        self.assertEqual(
            consolidated["independent_source_count"],
            2,
        )
        self.assertEqual(
            consolidated["direction_matches"],
            2,
        )

        strength = calculate_evidence_strength(consolidated)
        self.assertGreater(strength, 0.80)


if __name__ == "__main__":
    unittest.main()
