"""Tests for the investigation evidence layer."""

import unittest
from datetime import date

from src.investigation.evidence import (
    build_evidence_candidate,
    calculate_evidence_strength,
    consolidate_evidence,
    source_family,
    source_quality,
    infer_event_type,
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

    def test_common_publisher_aliases_share_a_family(self):
        self.assertEqual(
            source_family("https://www.livemint.com/example", "Mint"),
            "livemint",
        )
        self.assertEqual(
            source_family("https://www.livemint.com/example", "Livemint.com"),
            "livemint",
        )

    def test_yahoo_is_not_assumed_to_be_reuters(self):
        self.assertEqual(
            source_family(
                "https://finance.yahoo.com/example",
                "Yahoo Finance",
            ),
            "finance.yahoo.com",
        )

    def test_reuters_attribution_on_yahoo_is_grouped(self):
        self.assertEqual(
            source_family(
                "https://finance.yahoo.com/example",
                "Reuters",
            ),
            "reuters",
        )

    def test_distinctive_event_terms_beat_generic_financial_language(self):
        self.assertEqual(
            infer_event_type(
                "Maruti shares surge after GST tax cut. "
                "The company also reported quarterly profit and revenue."
            ),
            "policy_event",
        )

    def test_generic_profit_language_is_not_automatically_earnings(self):
        self.assertEqual(
            infer_event_type(
                "Shares rise after a policy announcement. "
                "The article compares last quarter profit and revenue."
            ),
            "policy_event",
        )

    def test_results_phrase_identifies_earnings(self):
        self.assertEqual(
            infer_event_type(
                "Company quarterly results beat analyst estimates."
            ),
            "earnings",
        )

    def test_block_trade_beats_background_earnings_language(self):
        self.assertEqual(
            infer_event_type(
                "Shares fall after a block deal. "
                "The article also discusses the company's profit."
            ),
            "block_trade",
        )

    def test_normalized_publisher_quality_is_consistent(self):
        self.assertEqual(source_quality("Mint"), 0.85)
        self.assertEqual(source_quality("Livemint.com"), 0.85)


if __name__ == "__main__":
    unittest.main()
