"""Run anomaly investigation through a configured search provider."""

from __future__ import annotations

from .investigator import investigate
from .queries import build_investigation_target


def investigate_with_provider(
    row: dict,
    provider,
    *,
    limit_per_query: int = 5,
) -> dict:
    """Search and investigate one anomaly without manual evidence input."""
    target = build_investigation_target(row)

    search_results = []
    for query in target["queries"]:
        search_results.extend(
            provider.search(
                query,
                limit=limit_per_query,
            )
        )

    result = investigate(target, search_results)
    result["queries"] = target["queries"]
    return result
