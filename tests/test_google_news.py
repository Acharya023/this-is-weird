"""Tests for Google News RSS normalization."""

import unittest

from src.investigation.google_news import GoogleNewsRSSProvider, _iso_date


class GoogleNewsTests(unittest.TestCase):

    def test_date_normalization(self):
        self.assertEqual(
            _iso_date("Mon, 28 Jul 2025 12:30:00 GMT"),
            "2025-07-28",
        )

    def test_provider_configuration(self):
        provider = GoogleNewsRSSProvider()
        self.assertEqual(
            provider.base_url,
            "https://news.google.com/rss/search",
        )


if __name__ == "__main__":
    unittest.main()
