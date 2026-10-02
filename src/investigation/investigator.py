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

    event_priority = {
        "earnings": 8,
        "guidance": 7,
        "block_trade": 7,
        "corporate_action": 6,
        "business_update": 5,
        "policy_event": 4,
        "analyst_research": 3,
        "sector_event": 2,
        "macro_event": 1,
        "unknown": 0,
    }

    event_candidates = [
        c for c in candidates
        if c.get("event_type") and c.get("event_type") != "unknown"
    ]
    inferred_event = (
        max(
            event_candidates,
            key=lambda c: (
                event_priority.get(c["event_type"], 0),
                c["evidence_relevance_score"],
            ),
        )["event_type"]
        if event_candidates
        else (best["event_type"] if best else None)
    )

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
        "best_event_type": inferred_event,
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
