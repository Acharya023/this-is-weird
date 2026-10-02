"""SearXNG search provider.

SearXNG exposes a JSON search endpoint when the instance has JSON output
enabled. The provider intentionally returns a small normalized result shape
that the investigation layer already understands.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urljoin

import requests


@dataclass
class SearXNGSearchProvider:
    """Search a SearXNG instance over its HTTP API."""

    base_url: str
    timeout: float = 15.0
    categories: str = "general,news"
    language: str = "en"
    safesearch: int = 0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        page: int = 1,
    ) -> list[dict[str, Any]]:
        """Search SearXNG and return normalized investigation results."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        response = requests.get(
            urljoin(self.base_url + "/", "search"),
            params={
                "q": query,
                "format": "json",
                "categories": self.categories,
                "language": self.language,
                "safesearch": self.safesearch,
                "pageno": page,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        payload = response.json()
        results = payload.get("results", [])

        normalized = []

        for result in results[:limit]:
            url = result.get("url")

            normalized.append(
                {
                    "title": result.get("title", ""),
                    "text": (
                        result.get("content")
                        or result.get("snippet")
                        or ""
                    ),
                    "published_date": _published_date(result),
                    "url": url,
                    "source_name": _source_name(result, url),
                    "event_type": None,
                }
            )

        return normalized


def _published_date(result: dict[str, Any]) -> str | None:
    """Extract a provider date when available."""
    published = result.get("publishedDate") or result.get("published_date")

    if isinstance(published, str):
        return published[:10]

    return None


def _source_name(
    result: dict[str, Any],
    url: str | None,
) -> str:
    """Prefer SearXNG's engine/source metadata, then hostname."""
    source = result.get("source")

    if source:
        return str(source)

    if not url:
        return "unknown"

    from urllib.parse import urlparse

    hostname = urlparse(url).netloc.lower()

    if hostname.startswith("www."):
        hostname = hostname[4:]

    return hostname


def search_queries(
    provider: SearXNGSearchProvider,
    queries: list[str],
    *,
    limit_per_query: int = 5,
) -> list[dict[str, Any]]:
    """Run multiple queries and combine their normalized results."""
    results = []

    for query in queries:
        results.extend(
            provider.search(
                query,
                limit=limit_per_query,
            )
        )

    return results
