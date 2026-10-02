"""Tests for the SearXNG provider normalization."""

import unittest

from src.investigation.searxng import SearXNGSearchProvider


class SearXNGProviderTests(unittest.TestCase):

    def test_base_url_is_normalized(self):
        provider = SearXNGSearchProvider(
            "https://example.com/"
        )
        self.assertEqual(
            provider.base_url,
            "https://example.com",
        )


if __name__ == "__main__":
    unittest.main()
