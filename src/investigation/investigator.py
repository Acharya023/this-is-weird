"""Provider-agnostic anomaly investigation orchestration."""

from .evidence import (
    build_event_groups,
    build_search_evidence,
    calculate_event_group_strength,
    consolidate_evidence,
)


def build_investigation_result(target: dict, candidates: list[dict]) -> dict:
    """Build one structured investigation record."""
    consolidated = consolidate_evidence(candidates)
    event_groups = build_event_groups(candidates)
    best_group = event_groups[0] if event_groups else None
    best = best_group["best_candidate"] if best_group else consolidated["best_candidate"]
    strength = calculate_event_group_strength(best_group)

    if not best_group:
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
        "independent_sources": best_group["independent_source_count"] if best_group else 0,
        "source_families": best_group["source_families"] if best_group else [],
        "direction_matches": best_group["direction_matches"] if best_group else 0,
        "direction_conflicts": best_group["direction_conflicts"] if best_group else 0,
        "evidence_strength": strength,
        "evidence_status": status,
        "best_event_type": best_group["event_type"] if best_group else None,
        "best_headline": best["headline"] if best else None,
        "best_source": best["source_name"] if best else None,
        "best_source_url": best["source_url"] if best else None,
        "best_evidence": best,
        "evidence_candidates": best_group["evidence_candidates"] if best_group else [],
        "event_groups": event_groups,
    }


def investigate(target: dict, search_results: list[dict]) -> dict:
    """Run the complete evidence pipeline for one target."""
    candidates = build_search_evidence(target, search_results)
    return build_investigation_result(target, candidates)
