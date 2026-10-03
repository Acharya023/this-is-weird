"""Benchmark financial relation precision/recall against a small gold set.

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

# Gold relations are deliberately limited to relations that are explicitly
# supported by the text. This benchmark measures extraction quality, not
# whether the model can infer unstated causal relationships.
SAMPLES = {
    "BHARTI_BLOCK_DEAL": {
        "text": (
            "Bharti Airtel shares decline after a large block deal. "
            "Singtel sold about 5.1 crore Bharti Airtel shares, putting pressure on the stock."
        ),
        "gold": {
            ("sells shares of", "Singtel", "Bharti Airtel shares"),
        },
    },
    "BAJAJ_GUIDANCE": {
        "text": (
            "Bajaj Finance falls after lowering AUM growth outlook. "
            "The company lowered its AUM growth guidance because of higher credit costs and competition."
        ),
        "gold": {
            ("lowers", "Bajaj Finance", "AUM growth outlook"),
            ("lowers", "Bajaj Finance", "AUM growth guidance"),
        },
    },
    "KOTAK_EARNINGS": {
        "text": (
            "Kotak Mahindra Bank shares sink after weaker-than-expected Q1 results. "
            "The bank reported margin pressure and asset-quality concerns."
        ),
        "gold": {
            ("reports", "The bank", "margin pressure and asset-quality concerns"),
        },
    },
    "MARUTI_GST_RALLY": {
        "text": (
            "Maruti Suzuki shares surge as proposed GST reform boosts auto stocks. "
            "The proposed tax reform helped lift shares across the automobile sector."
        ),
        "gold": set(),
    },
    "HCL_RESULTS": {
        "text": (
            "HCLTech reports quarterly results and gives FY26 growth guidance. "
            "Revenue growth guidance was set at 2% to 5% for FY26."
        ),
        "gold": {
            ("reports", "HCLTech", "quarterly results"),
            ("changes guidance for", "HCLTech", "FY26"),
        },
    },
    "TITAN_UPDATE": {
        "text": (
            "Titan reports a strong quarterly business update. "
            "Domestic jewellery sales grew 19% and consumer businesses also reported strong growth."
        ),
        "gold": {
            ("reports", "Titan", "strong quarterly business update"),
        },
    },
}


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def relation_key(relation_type: str, head: str, tail: str) -> tuple[str, str, str]:
    return (relation_type, normalize(head), normalize(tail))


def extract_relations(extractor, text: str, threshold: float):
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
            head = relation.get("head", {}).get("text", "")
            tail = relation.get("tail", {}).get("text", "")
            confidence = relation.get("confidence")

            if confidence is None:
                scores = [
                    float(score)
                    for score in (
                        relation.get("head", {}).get("confidence"),
                        relation.get("tail", {}).get("confidence"),
                    )
                    if score is not None
                ]
                confidence = min(scores) if scores else None

            rows.append(
                {
                    "key": relation_key(relation_type, head, tail),
                    "relation": relation_type,
                    "head": head,
                    "tail": tail,
                    "score": confidence,
                }
            )

    return rows


def match_gold(predicted, gold):
    """Match exact normalized relation triples."""
    predicted_keys = {row["key"] for row in predicted}
    gold_keys = {relation_key(*item) for item in gold}

    true_positive = predicted_keys & gold_keys
    false_positive = predicted_keys - gold_keys
    false_negative = gold_keys - predicted_keys

    return true_positive, false_positive, false_negative


def main() -> None:
    print("RELATION PRECISION / RECALL BENCHMARK")
    print(f"Model: {MODEL_NAME}")
    print(f"Thresholds: {THRESHOLDS}")
    print("No Google News requests are made.")
    print()

    load_start = time.perf_counter()
    extractor = AutoExtractor.from_pretrained(MODEL_NAME)
    print(f"MODEL LOAD TIME: {time.perf_counter() - load_start:.2f}s")
    print()

    summary = []

    for threshold in THRESHOLDS:
        tp = fp = fn = 0

        print("=" * 80)
        print(f"THRESHOLD {threshold:.2f}")

        for name, sample in SAMPLES.items():
            predicted = extract_relations(extractor, sample["text"], threshold)
            true_positive, false_positive, false_negative = match_gold(
                predicted,
                sample["gold"],
            )

            tp += len(true_positive)
            fp += len(false_positive)
            fn += len(false_negative)

            print(f"\n{name}")
            for row in predicted:
                marker = "TP" if row["key"] in true_positive else "FP"
                score = row["score"]
                score_text = f"{score:.3f}" if score is not None else "n/a"
                print(
                    f"  {marker} | {row['head']} -[{row['relation']}]-> "
                    f"{row['tail']} | score={score_text}"
                )

            for key in false_negative:
                relation, head, tail = key
                print(
                    f"  FN | {head} -[{relation}]-> {tail}"
                )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        summary.append((threshold, tp, fp, fn, precision, recall, f1))

        print(
            f"\nTHRESHOLD SUMMARY | TP={tp} FP={fp} FN={fn} "
            f"| precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}"
        )

    print("\n" + "=" * 80)
    print("OVERALL THRESHOLD COMPARISON")
    print("threshold | TP | FP | FN | precision | recall | f1")
    for threshold, tp, fp, fn, precision, recall, f1 in summary:
        print(
            f"{threshold:9.2f} | {tp:2d} | {fp:2d} | {fn:2d} "
            f"| {precision:.3f}     | {recall:.3f} | {f1:.3f}"
        )


if __name__ == "__main__":
    main()
