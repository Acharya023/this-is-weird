"""Benchmark search-result quality without running heavyweight NLP models."""

from __future__ import annotations

from datetime import date

from src.investigation.evidence import COMPANY_ALIASES, search_result_quality
from src.investigation.google_news import GoogleNewsRSSProvider
from src.investigation.queries import investigation_queries


TARGETS = [
    ("BHARTIARTL", date(2025, 11, 7)),
    ("BAJFINANCE", date(2025, 11, 11)),
    ("KOTAKBANK", date(2025, 7, 28)),
    ("MARUTI", date(2025, 8, 18)),
    ("HCLTECH", date(2025, 4, 23)),
    ("TITAN", date(2025, 10, 8)),
]


def main() -> None:
    provider = GoogleNewsRSSProvider()
    print("EVIDENCE SEARCH QUALITY BENCHMARK")
    print("Filters obvious listicle/price-only results before heavyweight NLP.")
    print()

    for symbol, target_date in TARGETS:
        print("=" * 100)
        print(f"{symbol} | {target_date.isoformat()}")

        seen = set()
        candidates = []
        for query in investigation_queries(symbol, target_date):
            try:
                results = provider.search(query, limit=10)
            except Exception as exc:
                print(f"SEARCH ERROR | {type(exc).__name__}: {exc}")
                continue

            for article in results:
                url = article.get("url")
                if not url or url in seen:
                    continue
                if str(article.get("published_date") or "") != target_date.isoformat():
                    continue
                seen.add(url)
                candidates.append(article)

        ranked = sorted(
            candidates,
            key=lambda article: search_result_quality(
                article,
                COMPANY_ALIASES.get(symbol, [symbol]),
            ),
            reverse=True,
        )

        print(f"RAW CANDIDATES: {len(candidates)}")
        for article in ranked[:8]:
            score = search_result_quality(
                article,
                COMPANY_ALIASES.get(symbol, [symbol]),
            )
            print(f"QUALITY {score:.2f} | {article.get('source_name')} | {article.get('title')}")

        print()


if __name__ == "__main__":
    main()
