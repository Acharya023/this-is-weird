"""Tests for the investigation evidence layer."""

import unittest
from datetime import date

from src.investigation.evidence import (
    build_evidence_candidate,
    build_event_groups,
    calculate_evidence_strength,
    consolidate_evidence,
    source_family,
    source_quality,
    infer_event_type,
    entity_relevance,
)


class EvidenceTests(unittest.TestCase):

    def test_same_day_matching_event_scores_strongly(self):
        candidate = build_evidence_candidate(
            "MARUTI", date(2025, 8, 18), 0.0875, date(2025, 8, 18),
            "policy_event", "Maruti shares rally on GST cuts",
            "Auto stocks rose sharply.", "Reuters",
            "https://www.reuters.com/example",
        )
        self.assertTrue(candidate["direction_matches"])
        self.assertGreater(candidate["evidence_relevance_score"], 0.85)

    def test_reuters_syndication_is_one_source_family(self):
        candidates = [
            build_evidence_candidate("MARUTI", date(2025, 8, 18), 0.08, date(2025, 8, 18), "policy_event", "Maruti rallies", "GST reform boosts autos", "Reuters", "https://www.reuters.com/example"),
            build_evidence_candidate("MARUTI", date(2025, 8, 18), 0.08, date(2025, 8, 18), "policy_event", "Maruti rallies", "GST reform boosts autos", "Yahoo Finance", "https://finance.yahoo.com/example"),
            build_evidence_candidate("MARUTI", date(2025, 8, 18), 0.08, date(2025, 8, 18), "policy_event", "Auto stocks rally", "GST reform boosts autos", "Moneycontrol", "https://www.moneycontrol.com/example"),
        ]
        consolidated = consolidate_evidence(candidates)
        self.assertEqual(consolidated["independent_source_count"], 2)
        self.assertEqual(consolidated["direction_matches"], 2)
        self.assertGreater(calculate_evidence_strength(consolidated), 0.80)

    def test_common_publisher_aliases_share_a_family(self):
        self.assertEqual(source_family("https://www.livemint.com/example", "Mint"), "livemint")
        self.assertEqual(source_family("https://www.livemint.com/example", "Livemint.com"), "livemint")

    def test_yahoo_is_not_assumed_to_be_reuters(self):
        self.assertEqual(source_family("https://finance.yahoo.com/example", "Yahoo Finance"), "finance.yahoo.com")

    def test_reuters_attribution_on_yahoo_is_grouped(self):
        self.assertEqual(source_family("https://finance.yahoo.com/example", "Reuters"), "reuters")

    def test_distinctive_event_terms_beat_generic_financial_language(self):
        self.assertEqual(infer_event_type("Maruti shares surge after GST tax cut. The company also reported quarterly profit and revenue."), "policy_event")

    def test_generic_profit_language_is_not_automatically_earnings(self):
        self.assertEqual(infer_event_type("Shares rise after a policy announcement. The article compares last quarter profit and revenue."), "policy_event")

    def test_results_phrase_identifies_earnings(self):
        self.assertEqual(infer_event_type("Company quarterly results beat analyst estimates."), "earnings")

    def test_block_trade_beats_background_earnings_language(self):
        self.assertEqual(infer_event_type("Shares fall after a block deal. The article also discusses the company's profit."), "block_trade")

    def test_normalized_publisher_quality_is_consistent(self):
        self.assertEqual(source_quality("Mint"), 0.85)
        self.assertEqual(source_quality("Livemint.com"), 0.85)


    def test_unrelated_short_symbol_result_has_no_entity_relevance(self):
        self.assertEqual(
            entity_relevance(
                "LT",
                "Impact of local therapy in metastatic renal cell carcinoma",
                "A retrospective medical analysis.",
            ),
            0.0,
        )

    def test_company_name_matches_entity(self):
        self.assertEqual(
            entity_relevance(
                "LT",
                "Larsen & Toubro shares rise after strong results",
                "",
            ),
            1.0,
        )

    def test_unrelated_results_are_removed_from_search_evidence(self):
        from src.investigation.evidence import build_search_evidence

        candidates = build_search_evidence(
            {
                "symbol": "LT",
                "anomaly_date": date(2025, 7, 30),
                "daily_return": 0.048,
            },
            [
                {
                    "title": "Impact of local therapy in metastatic renal cell carcinoma",
                    "text": "A retrospective medical analysis.",
                    "published_date": "2025-07-30",
                    "url": "https://www.nature.com/example",
                    "source_name": "Nature",
                    "event_type": None,
                },
                {
                    "title": "Larsen & Toubro shares rise after Q1 results",
                    "text": "L&T reported strong quarterly results.",
                    "published_date": "2025-07-30",
                    "url": "https://www.reuters.com/example",
                    "source_name": "Reuters",
                    "event_type": None,
                },
            ],
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["source_name"], "Reuters")

    def test_event_groups_choose_catalyst(self):
        candidates = [
            build_evidence_candidate("MARUTI", date(2025, 8, 18), 0.08, date(2025, 8, 18), "policy_event", "Maruti rallies on GST reform", "GST cuts boost autos", "Reuters", "https://www.reuters.com/a"),
            build_evidence_candidate("MARUTI", date(2025, 8, 18), 0.08, date(2025, 8, 18), "policy_event", "Auto stocks jump after tax cut", "GST reform boosts demand", "Moneycontrol", "https://www.moneycontrol.com/b"),
            build_evidence_candidate("MARUTI", date(2025, 8, 18), 0.08, date(2025, 8, 18), "earnings", "Maruti profit update", "Quarterly profit is discussed", "Economic Times", "https://economictimes.indiatimes.com/c"),
        ]
        groups = build_event_groups(candidates)
        self.assertEqual(groups[0]["event_type"], "policy_event")
        self.assertEqual(groups[0]["independent_source_count"], 2)

    def test_specific_catalyst_beats_many_generic_articles(self):
        candidates = [
            *[
                build_evidence_candidate(
                    "KOTAKBANK", date(2025, 7, 28), -0.07,
                    date(2025, 7, 28), "unknown",
                    f"Kotak shares fall article {i}",
                    "Markets were weak and shares declined.",
                    f"Generic Publisher {i}",
                    f"https://generic{i}.example/article",
                )
                for i in range(10)
            ],
            build_evidence_candidate(
                "KOTAKBANK", date(2025, 7, 28), -0.07,
                date(2025, 7, 28), "earnings",
                "Kotak Q1 results disappoint",
                "Quarterly results missed expectations.",
                "Livemint",
                "https://www.livemint.com/example",
            ),
            build_evidence_candidate(
                "KOTAKBANK", date(2025, 7, 28), -0.07,
                date(2025, 7, 28), "earnings",
                "Kotak profit falls",
                "Weak quarterly results pressure the stock.",
                "Reuters",
                "https://www.reuters.com/example",
            ),
        ]
        groups = build_event_groups(candidates)
        self.assertEqual(groups[0]["event_type"], "earnings")
        self.assertGreater(groups[0]["score"], groups[1]["score"])

    def test_event_group_score_is_capped(self):
        candidates = [
            build_evidence_candidate(
                "KOTAKBANK", date(2025, 7, 28), -0.07, date(2025, 7, 28),
                "earnings", "Kotak earnings", "weak quarterly results",
                "Publisher" + str(i), "https://publisher" + str(i) + ".example/a",
            )
            for i in range(10)
        ]
        groups = build_event_groups(candidates)
        self.assertEqual(groups[0]["event_type"], "earnings")
        self.assertLessEqual(groups[0]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
