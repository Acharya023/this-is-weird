"""Calibrate GLiNER2 relation extraction thresholds on financial examples.

Standalone benchmark. It does not change production investigation logic.
"""

from __future__ import annotations

import time

from gliner2 import AutoExtractor


MODEL_NAME = "fastino/gliner2.5-small-v1"

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

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]

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


def relation_score(relation: dict) -> float | None:
    confidence = relation.get("confidence")
    if confidence is not None:
        return float(confidence)

    scores = [
        float(score)
        for score in (
            relation.get("head", {}).get("confidence"),
            relation.get("tail", {}).get("confidence"),
        )
        if score is not None
    ]
    return min(scores) if scores else None


def extract_at_threshold(extractor, text: str, threshold: float):
    result = extractor.extract_relations(
        text,
        RELATIONS,
        threshold=threshold,
        include_spans=True,
        include_confidence=True,
    )

    rows = []
    for relation_type, relations in result.get("relation_extraction", {}).items():
        for relation in relations:
            rows.append(
                (
                    relation_type,
                    relation.get("head", {}).get("text", ""),
                    relation.get("tail", {}).get("text", ""),
                    relation_score(relation),
                )
            )
    return rows


def main() -> None:
    print("RELATION THRESHOLD CALIBRATION")
    print(f"Model: {MODEL_NAME}")
    print(f"Thresholds: {THRESHOLDS}")
    print("No Google News requests are made.")
    print()

    load_start = time.perf_counter()
    extractor = AutoExtractor.from_pretrained(MODEL_NAME)
    print(f"MODEL LOAD TIME: {time.perf_counter() - load_start:.2f}s")
    print()

    total = 0.0

    for name, text in SAMPLES.items():
        print("=" * 80)
        print(name)

        for threshold in THRESHOLDS:
            start = time.perf_counter()
            rows = extract_at_threshold(extractor, text, threshold)
            elapsed = time.perf_counter() - start
            total += elapsed

            print(f"THRESHOLD {threshold:.2f} | time={elapsed:.3f}s")

            if not rows:
                print("  none")
                continue

            for relation_type, head, tail, score in rows:
                score_text = f"{score:.3f}" if score is not None else "n/a"
                print(
                    f"  {head} -[{relation_type}]-> {tail} "
                    f"| score={score_text}"
                )

        print()

    print("=" * 80)
    print("RUNTIME SUMMARY")
    print(f"ALL RELATION CALLS total={total:.2f}s")
    print(f"ALL RELATION CALLS avg={total / (len(SAMPLES) * len(THRESHOLDS)):.2f}s")


if __name__ == "__main__":
    main()
