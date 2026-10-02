"""Google News RSS search provider.

Google News exposes keyword-search RSS feeds. This provider normalizes feed
items into the same shape consumed by the investigation evidence layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests


@dataclass
class GoogleNewsRSSProvider:
    """Search Google News through its RSS search feed."""

    language: str = "en-IN"
    country: str = "IN"
    timeout: float = 15.0

    @property
    def base_url(self) -> str:
        return "https://news.google.com/rss/search"

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return normalized Google News search results."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        ceid = f"{self.country}:{self.language.split('-')[0]}"

        response = requests.get(
            self.base_url,
            params={
                "q": query,
                "hl": self.language,
                "gl": self.country,
                "ceid": ceid,
            },
            headers={
                "User-Agent": "ThisIsWeird/0.1 research",
                "Accept": "application/rss+xml, application/xml, text/xml",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)
        items = root.findall("./channel/item")

        normalized = []

        for item in items[:limit]:
            title = _text(item, "title")
            description = _text(item, "description")
            link = _text(item, "link")
            pub_date = _text(item, "pubDate")
            source = item.find("source")

            normalized.append(
                {
                    "title": title,
                    "text": description,
                    "published_date": _iso_date(pub_date),
                    "url": link,
                    "source_name": (
                        source.text.strip()
                        if source is not None and source.text
                        else "Google News"
                    ),
                    "event_type": None,
                }
            )

        return normalized


def _text(parent: ET.Element, tag: str) -> str:
    element = parent.find(tag)
    return (
        element.text.strip()
        if element is not None and element.text
        else ""
    )


def _iso_date(value: str) -> str | None:
    if not value:
        return None

    try:
        return parsedate_to_datetime(value).astimezone(
            timezone.utc
        ).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        return None
