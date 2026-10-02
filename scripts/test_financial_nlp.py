""""Fast, reproducible financial NLP benchmark."""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.investigation.semantic import FinancialNLPAnalyzer


SAMPLES = [
    {
        "name": "MARUTI_GST_RALLY",
        "symbol": "MARUTI",
        "title": "Maruti Suzuki shares surge as proposed GST reform boosts auto stocks",
        "text": "Indian auto stocks rose after the government proposed changes to GST rates. Maruti Suzuki shares jumped nearly 9%.",
    },
    {
        "name": "KOTAK_EARNINGS",
        "symbol": "KOTAKBANK",
        "title": "Kotak Mahindra Bank shares sink after weaker-than-expected Q1 results",
        "text": "Kotak Mahindra Bank reported quarterly results that raised asset-quality concerns, while margins remained under pressure.",
    },
    {
        "name": "BHARTI_BLOCK_DEAL",
        "symbol": "BHARTIARTL",
        "title": "Bharti Airtel shares decline after large block deal",
        "text": "Singtel sold about 5.1 crore Bharti Airtel shares in a block trade. The stock fell around 3.5%.",
    },
    {
        "name": "BAJAJ_GUIDANCE",
        "symbol": "BAJFINANCE",
        "title": "Bajaj Finance falls after lowering AUM growth outlook",
        "text": "Bajaj Finance cut its AUM growth forecast because of rising competition and higher credit costs.",
    },
    {
        "name": "HCL_RESULTS",
        "symbol": "HCLTECH",
        "title": "HCLTech reports quarterly results and gives FY26 growth guidance",
        "text": "HCLTech reported quarterly revenue below expectations and guided for FY26 growth of 2% to 5%.",
    },
    {
        "name": "TITAN_UPDATE",
        "symbol": "TITAN",
        "title": "Titan reports strong quarterly business update",
        "text": "Titan said domestic jewellery sales grew 19% in the quarter, while its consumer businesses also reported growth.",
    },
]


def print_entities(entities):
    for entity in entities:
        print(
            f"  ENTITY | {entity.get('label'):15} | "
            f"{entity.get('text')} | score={entity.get('score', 0):.3f}"
        )


def print_facts(facts):
    for fact in facts:
        print(
            f"  FACT   | {fact['label']:15} | {fact['text']} "
            f"| score={fact['score']:.3f}"
        )


def print_events(events):
    if not events:
        print("  EVENT  | none above threshold")
        return
    for event in events:
        print(
            f"  EVENT  | {event['label']:20} | "
            f"{event['mapped_event_type']:20} | score={event['score']:.3f}"
        )


def main():
    print("SMALL FINANCIAL NLP TEST")
    print(f"Samples: {len(SAMPLES)}")
    print("No Google News requests are made.")
    print()

    timings = {}
    analyzer = FinancialNLPAnalyzer(enable_sentiment=True)

    for sample in SAMPLES:
        print("\n" + "=" * 90)
        print(f"{sample['name']} | {sample['symbol']}")
        print(sample["title"])
        print("=" * 90)

        text = analyzer.article_text(sample)

        start = time.perf_counter()
        entities = analyzer.extract_entities(text)
        elapsed = time.perf_counter() - start
        timings.setdefault("NER", []).append(elapsed)
        print(f"Company NER time: {elapsed:.2f}s")
        print_entities(entities)

        start = time.perf_counter()
        facts = analyzer.extract_financial_facts(text)
        elapsed = time.perf_counter() - start
        timings.setdefault("FACTS", []).append(elapsed)
        print(f"Financial facts time: {elapsed:.4f}s")
        print_facts(facts)

        start = time.perf_counter()
        events = analyzer.detect_events(text)
        elapsed = time.perf_counter() - start
        timings.setdefault("EVENT", []).append(elapsed)
        print(f"Event model time: {elapsed:.2f}s")
        print_events(events)

        start = time.perf_counter()
        result = analyzer._load_sentiment()(text)[0]
        elapsed = time.perf_counter() - start
        timings.setdefault("SENTIMENT", []).append(elapsed)
        print(
            f"FinBERT: {result['label']} "
            f"(score={float(result['score']):.3f}, time={elapsed:.2f}s)"
        )

    print("\n" + "=" * 90)
    print("RUNTIME SUMMARY")
    print("=" * 90)
    for stage, values in timings.items():
        print(
            f"{stage:10} total={sum(values):.2f}s "
            f"avg={sum(values) / len(values):.2f}s"
        )
    print("\nThe large GLiNER dependency is no longer required.")
    print("Hugging Face model weights remain cached after the first download.")


if __name__ == "__main__":
    main()
"