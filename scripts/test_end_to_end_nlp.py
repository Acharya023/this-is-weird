"""End-to-end NLP benchmark on real Google News investigation evidence.

This benchmark does not change production investigation scoring. It searches
real evidence for representative anomalies, then runs the complete NLP layer:
company NER, deterministic financial facts, FinDeBERTa event signals,
GLiNER2 relations, and FinBERT sentiment.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from datetime import date

from src.investigation.evidence import COMPANY_ALIASES
from src.investigation.google_news import GoogleNewsRSSProvider
from src.investigation.queries import investigation_queries
from src.investigation.semantic import FinancialNLPAnalyzer


TARGETS = [
    ("BHARTIARTL", date(2025, 11, 7)),
    ("BAJFINANCE", date(2025, 11, 11)),
    ("KOTAKBANK", date(2025, 7, 28)),
    ("MARUTI", date(2025, 8, 18)),
    ("HCLTECH", date(2025, 4, 23)),
    ("TITAN", date(2025, 10, 8)),
]

ARTICLES_PER_TARGET = 3


def article_date_ok(article: dict, target_date: date) -> bool:
    published = article.get("published_date")
    return not published or str(published) == target_date.isoformat()


def main() -> None:
    print("END-TO-END FINANCIAL NLP BENCHMARK")
    print(f"Targets: {len(TARGETS)}")
    print(f"Articles per target: {ARTICLES_PER_TARGET}")
    print("Uses real Google News RSS. No production scoring changes.")
    print()

    provider = GoogleNewsRSSProvider()
    analyzer = FinancialNLPAnalyzer()

    searched = 0
    analyzed = 0
    seen_urls: set[str] = set()
    total_nlp_time = 0.0

    for symbol, target_date in TARGETS:
        print("=" * 100)
        print(f"{symbol} | {target_date.isoformat()}")

        articles = []
        for query in investigation_queries(symbol, target_date):
            try:
                results = provider.search(query, limit=5)
            except Exception as exc:
                print(f"SEARCH ERROR | {type(exc).__name__}: {exc}")
                continue

            searched += len(results)
            for article in results:
                url = article.get("url")
                if not url or url in seen_urls:
                    continue
                if not article_date_ok(article, target_date):
                    continue
                seen_urls.add(url)
                articles.append(article)
                if len(articles) >= ARTICLES_PER_TARGET:
                    break

            if len(articles) >= ARTICLES_PER_TARGET:
                break

        print(f"Articles selected: {len(articles)}")

        for index, article in enumerate(articles, 1):
            title = article.get("title", "")
            source = article.get("source_name", "")
            print()
            print(f"ARTICLE {index} | {source}")
            print(title)

            start = time.perf_counter()
            analysis = analyzer.analyze(article, target_aliases=COMPANY_ALIASES.get(symbol, [symbol]))
            elapsed = time.perf_counter() - start
            total_nlp_time += elapsed
            analyzed += 1

            companies = [
                item for item in analysis.entities
                if item.get("label") == "company"
            ]

            print(f"NLP time: {elapsed:.2f}s")
            print(f"DETERMINISTIC EVENT | {analysis.deterministic_event_type}")
            print(
                "TARGET COMPANIES | "
                + (
                    ", ".join(f"{item['text']} ({item['score']:.2f})" for item in analysis.target_entities)
                    if analysis.target_entities else "none"
                )
            )
            print(
                "COMPANIES | "
                + (
                    ", ".join(
                        f"{item['text']} ({item['score']:.2f})"
                        for item in companies
                    )
                    if companies
                    else "none"
                )
            )

            print(
                "EVENT SIGNALS | "
                + (
                    ", ".join(
                        item["text"]
                        for item in analysis.financial_facts
                        if item["label"] == "event_signal"
                    )
                    or "none"
                )
            )

            print(
                "FINANCIAL TERMS | "
                + (
                    ", ".join(
                        item["text"]
                        for item in analysis.financial_facts
                        if item["label"] == "financial_term"
                    )
                    or "none"
                )
            )

            print(
                "FACTS | "
                + (
                    ", ".join(
                        f"{item['label']}={item['text']}"
                        for item in analysis.financial_facts
                        if item["label"] in {"percentage", "amount", "quantity", "date"}
                    )
                    or "none"
                )
            )

            print(
                "EVENT MODEL | "
                + (
                    ", ".join(
                        f"{item['label']} ({item['score']:.2f})"
                        for item in analysis.event_labels[:4]
                    )
                    or "none"
                )
            )

            print(
                "RELATIONS | "
                + (
                    "; ".join(
                        f"{item['head']} -[{item['relation']}]-> "
                        f"{item['tail']} ({item['score']:.2f})"
                        if item["score"] is not None
                        else f"{item['head']} -[{item['relation']}]-> {item['tail']}"
                        for item in analysis.relations[:5]
                    )
                    or "none"
                )
            )

            print(
                f"SENTIMENT | {analysis.sentiment or 'none'} "
                f"({analysis.sentiment_confidence:.2f})"
                if analysis.sentiment_confidence is not None
                else "SENTIMENT | none"
            )

    print()
    print("=" * 100)
    print("BENCHMARK SUMMARY")
    print(f"RAW SEARCH RESULTS: {searched}")
    print(f"ARTICLES ANALYZED: {analyzed}")
    print(
        f"NLP TOTAL: {total_nlp_time:.2f}s "
        f"AVG: {total_nlp_time / analyzed:.2f}s"
        if analyzed
        else "NLP TOTAL: 0.00s"
    )


if __name__ == "__main__":
    main()
