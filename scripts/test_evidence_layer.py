"""Focused benchmark for the target-aware deterministic evidence layer."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.investigation.evidence import COMPANY_ALIASES
from src.investigation.semantic import FinancialNLPAnalyzer


def main() -> None:
    analyzer = FinancialNLPAnalyzer(
        ner_model=type("FakeNER", (), {
            "__call__": lambda self, text: [
                {"word": "Bharti Airtel", "entity_group": "company", "score": 0.97},
                {"word": "Bharti", "entity_group": "company", "score": 0.80},
                {"word": "Airtel", "entity_group": "company", "score": 0.75},
                {"word": "Singtel", "entity_group": "organization", "score": 0.91},
            ]
        })(),
        event_model=type("FakeEvent", (), {})(),
        sentiment_classifier=type("FakeSentiment", (), {
            "__call__": lambda self, text: [{"label": "negative", "score": 0.95}]
        })(),
    )

    article = {
        "title": "Bharti Airtel shares decline 3.5% after 5.1 crore shares change hands via block deal",
        "text": "Singtel sold about 5.1 crore Bharti Airtel shares in a block trade.",
    }

    text = analyzer.article_text(article)
    facts = analyzer.extract_financial_facts(text)
    target = analyzer.target_company_entities(
        analyzer.extract_entities(text),
        COMPANY_ALIASES["BHARTIARTL"],
    )
    event = analyzer.deterministic_event_type(text)

    print("EVIDENCE LAYER TEST")
    print("TARGET ENTITIES:", target)
    print("FACTS:", facts)
    print("EVENT:", event)

    fact_text = {(item["label"], item["text"]) for item in facts}
    assert ("percentage", "3.5%") in fact_text
    assert ("quantity", "5.1 crore shares") in fact_text
    assert ("amount", "5.1 crore") not in fact_text
    assert event == "block_trade"
    assert len(target) == 1
    assert target[0]["text"] == "Bharti Airtel"
    assert all(item["text"].strip() for item in target)

    print("PASS")


if __name__ == "__main__":
    main()
