"""Reproducible semantic investigation test.

Run this from the repository root. It downloads the Hugging Face models
on first use and applies them to real historical Google News results.
"""

from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.investigation.google_news import GoogleNewsRSSProvider
from src.investigation.queries import investigation_queries
from src.investigation.semantic import SemanticEvidenceAnalyzer


def main():
    targets = [
        ("MARUTI", date(2025, 8, 18)),
        ("KOTAKBANK", date(2025, 7, 28)),
        ("BHARTIARTL", date(2025, 11, 7)),
    ]

    provider = GoogleNewsRSSProvider()
    analyzer = SemanticEvidenceAnalyzer()

    for symbol, event_date in targets:
        print("\n" + "=" * 90)
        print(f"{symbol} | {event_date}")
        print("=" * 90)

        results = []
        for query in investigation_queries(symbol, event_date):
            results.extend(provider.search(query, limit=5))

        seen = set()
        unique = []
        for result in results:
            url = result.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            unique.append(result)

        for article in unique[:8]:
            analysis = analyzer.analyze(article)
            print(f"\nTITLE: {article.get('title', '')}")
            print(f"SOURCE: {article.get('source_name', '')}")
            print(
                "SEMANTIC EVENT: "
                f"{analysis.event_type} "
                f"({analysis.event_confidence:.3f})"
            )
            print(f"EVENT LABEL: {analysis.event_label}")
            print(
                "FINBERT SENTIMENT: "
                f"{analysis.sentiment} "
                f"({analysis.sentiment_confidence:.3f})"
            )


if __name__ == "__main__":
    main()
