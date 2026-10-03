"""Benchmark semantic relation extraction for financial investigation evidence.

This is a standalone benchmark. It does not change production investigation logic.
It uses the compact GLiNER2.5 Small checkpoint to test whether schema-driven
relations can capture actor -> action, event -> company, reason -> consequence,
and market-effect relationships in representative financial text.
"""

from __future__ import annotations

import time

from gliner2 import AutoExtractor


MODEL_NAME = "fastino/gliner2.5-small-v1"

CONTROL_RELATIONS = ["works_for"]

RELATIONS = [
    "sells shares of",
    "causes",
    "affects",
    "reports",
    "announces",
    "results in",
    "is the reason for",
    "lowers",
    "raises",
    "changes guidance for",
]

SAMPLES = {
    "BHARTI_BLOCK_DEAL": (
        "Bharti Airtel shares decline after a large block deal. "
        "Singtel sold about 5.1 crore Bharti Airtel shares, putting pressure on the stock."
    ),
    "BAJAJ_GUIDANCE": (
        "Bajaj Finance falls after lowering AUM growth outlook. "
        "The company lowered its AUM growth guidance because of higher credit costs and competition."
    ),
    "KOTAK_EARNINGS": (
        "Kotak Mahindra Bank shares sink after weaker-than-expected Q1 results. "
        "The bank reported margin pressure and asset-quality concerns."
    ),
    "MARUTI_GST_RALLY": (
        "Maruti Suzuki shares surge as proposed GST reform boosts auto stocks. "
        "The proposed tax reform helped lift shares across the automobile sector."
    ),
    "HCL_RESULTS": (
        "HCLTech reports quarterly results and gives FY26 growth guidance. "
        "Revenue growth guidance was set at 2% to 5% for FY26."
    ),
    "TITAN_UPDATE": (
        "Titan reports a strong quarterly business update. "
        "Domestic jewellery sales grew 19% and consumer businesses also reported strong growth."
    ),
}


def main() -> None:
    print("RELATION / CAUSAL EXTRACTION TEST")
    print(f"Model: {MODEL_NAME}")
    print("No Google News requests are made.")
    print()

    load_start = time.perf_counter()
    extractor = AutoExtractor.from_pretrained(MODEL_NAME)
    print(f"MODEL LOAD TIME: {time.perf_counter() - load_start:.2f}s")
    print()

    total = 0.0

    print("=" * 80)
    print("API / MODEL CONTROL")
    control_text = "Alice works for Acme."
    print(control_text)
    start = time.perf_counter()
    control = extractor.extract_relations(
        control_text,
        CONTROL_RELATIONS,
        threshold=0.30,
        include_spans=True,
        include_confidence=True,
    )
    elapsed = time.perf_counter() - start
    print(f"Control extraction time: {elapsed:.3f}s")
    print(control)
    print()

    for name, text in SAMPLES.items():
        print("=" * 80)
        print(name)
        print(text)
        print()

        start = time.perf_counter()
        result = extractor.extract_relations(
            text,
            RELATIONS,
            threshold=0.30,
            include_spans=True,
            include_confidence=True,
        )
        elapsed = time.perf_counter() - start
        total += elapsed

        print(f"Relation extraction time: {elapsed:.3f}s")

        relation_data = result.get("relation_extraction", {})
        printed = 0
        for relation_type, relations in relation_data.items():
            for relation in relations:
                head = relation.get("head", {})
                tail = relation.get("tail", {})
                confidence = relation.get("confidence")
                if confidence is None:
                    head_score = head.get("confidence")
                    tail_score = tail.get("confidence")
                    scores = [
                        float(score)
                        for score in (head_score, tail_score)
                        if score is not None
                    ]
                    confidence = min(scores) if scores else None

                score_text = (
                    f"{float(confidence):.3f}"
                    if confidence is not None
                    else "n/a"
                )
                print(
                    "REL | "
                    f"{head.get('text', '')} "
                    f"-[{relation_type}]-> "
                    f"{tail.get('text', '')} "
                    f"| score={score_text}"
                )
                printed += 1

        if not printed:
            print("REL | none above model threshold")

        print()

    print("=" * 80)
    print("RUNTIME SUMMARY")
    print(f"RELATION total={total:.2f}s avg={total / len(SAMPLES):.2f}s")


if __name__ == "__main__":
    main()
