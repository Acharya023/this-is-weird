"""Provider-agnostic anomaly investigation orchestration."""

from .evidence import (
    build_search_evidence,
    calculate_evidence_strength,
    consolidate_evidence,
)


def build_investigation_result(
    target: dict,
    candidates: list[dict],
) -> dict:
    """Build one structured investigation record."""
    consolidated = consolidate_evidence(candidates)
    strength = calculate_evidence_strength(consolidated)
    best = consolidated["best_candidate"]

    if not best:
        status = "unclear"
    elif strength >= 0.80:
        status = "strong"
    elif strength >= 0.60:
        status = "moderate"
    elif strength >= 0.40:
        status = "weak"
    else:
        status = "unclear"

    return {
        "date": target["anomaly_date"],
        "symbol": target["symbol"],
        "search_candidates": consolidated["candidate_count"],
        "independent_sources": consolidated["independent_source_count"],
        "source_families": consolidated["source_families"],
        "direction_matches": consolidated["direction_matches"],
        "direction_conflicts": consolidated["direction_conflicts"],
        "evidence_strength": strength,
        "evidence_status": status,
        "best_event_type": best["event_type"] if best else None,
        "best_headline": best["headline"] if best else None,
        "best_source": best["source_name"] if best else None,
        "best_source_url": best["source_url"] if best else None,
    }


def investigate(target: dict, search_results: list[dict]) -> dict:
    """Run the complete evidence pipeline for one target."""
    candidates = build_search_evidence(
        target,
        search_results,
    )

    return build_investigation_result(
        target,
        candidates,
    )
