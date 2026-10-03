"""Focused regression tests for financial NLP evidence postprocessing."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.investigation.semantic import FinancialNLPAnalyzer


class FakeNER:
    def predict_entities(self, text, labels, threshold=0.5):
        return [
            {"text": "Bharti Airtel", "label": "company", "score": 0.97},
            {"text": "Airtel", "label": "company", "score": 0.73},
            {"text": "Singtel", "label": "organization", "score": 0.91},
        ]


def main() -> None:
    analyzer = FinancialNLPAnalyzer(
        ner_model=FakeNER(),
        enable_sentiment=False,
    )

    article = analyzer.article_text({
        "title": "Bharti Airtel shares decline 3.5% after 5.1 crore shares change hands via block deal",
        "text": "Singtel sold about 5.1 crore Bharti Airtel shares in a block trade.",
    })

    target = analyzer.target_company_entities(
        analyzer.extract_entities(article),
        ["bharti airtel", "airtel"],
    )
    facts = analyzer.extract_financial_facts(article)

    print("NLP POSTPROCESSING TEST")
    print("TARGET ENTITIES:", target)
    print("FACTS:", facts)

    assert target[0]["text"] == "Bharti Airtel"
    assert "Airtel" not in {item["text"] for item in target}
    assert any(
        item["label"] == "quantity" and item["text"].lower() == "5.1 crore shares"
        for item in facts
    )
    assert not any(
        item["label"] == "amount" and item["text"].lower() == "5.1 crore"
        for item in facts
    )

    print("PASS")


if __name__ == "__main__":
    main()
