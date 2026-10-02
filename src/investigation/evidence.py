"""Evidence normalization, scoring and corroboration."""

from urllib.parse import urlparse


DOMAIN_ALIASES = {
    "finance.yahoo.com": "reuters",
    "sg.finance.yahoo.com": "reuters",
    "uk.finance.yahoo.com": "reuters",
    "ca.finance.yahoo.com": "reuters",
    "in.finance.yahoo.com": "reuters",
}

PUBLISHER_ALIASES = {
    "mint": "livemint",
    "livemint.com": "livemint",
    "the mint": "livemint",
    "economic times": "economic times",
    "the economic times": "economic times",
    "financial express": "financial express",
    "the financial express": "financial express",
    "business standard": "business standard",
    "moneycontrol": "moneycontrol",
}


def source_domain(url: str | None) -> str:
    if not url:
        return "unknown"
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return DOMAIN_ALIASES.get(domain, domain)


def normalize_publisher_name(source_name: str | None) -> str:
    """Normalize common publisher-name variants before grouping evidence."""
    if not source_name:
        return "unknown"

    name = " ".join(source_name.strip().lower().split())
    return PUBLISHER_ALIASES.get(name, name)


def source_family(
    url: str | None,
    source_name: str | None = None,
) -> str:
    """Collapse known syndication and publisher aliases into evidence families."""
    domain = source_domain(url)

    if domain == "news.google.com" and source_name:
        return normalize_publisher_name(source_name)

    if domain in {"reuters.com", "reuters"}:
        return "reuters"

    return normalize_publisher_name(domain)


def classify_event_direction(text: str) -> str:
    text = text.lower()

    positive_terms = [
        "surge", "rally", "rose", "gains", "higher",
        "growth", "upgrade", "strong", "beats", "benefit",
        "cut taxes", "positive",
    ]
    negative_terms = [
        "fall", "fell", "decline", "drop", "tumble",
        "lower", "downgrade", "miss", "weak", "pressure",
        "cut forecast", "negative",
    ]

    positive = sum(term in text for term in positive_terms)
    negative = sum(term in text for term in negative_terms)

    if positive > negative:
        return "positive"
    if negative > positive:
        return "negative"
    return "neutral"


def infer_event_type(text: str) -> str:
    text = text.lower()

    if any(x in text for x in [
        "quarterly results", "quarterly result", "q1", "q2", "q3", "q4",
        "profit", "revenue", "earnings", "net interest income",
        "net interest margin", "nim", "provisions", "bad loans",
    ]):
        return "earnings"
    if any(x in text for x in [
        "stake sale", "block deal", "bulk deal",
    ]):
        return "block_trade"
    if any(x in text for x in [
        "gst", "tax cut", "government policy", "policy reform",
    ]):
        return "policy_event"
    if any(x in text for x in [
        "guidance", "forecast", "outlook",
    ]):
        return "guidance"
    if any(x in text for x in [
        "demerger", "bonus", "split", "dividend",
        "rights issue", "buyback",
    ]):
        return "corporate_action"
    if any(x in text for x in [
        "business update", "sales update", "operational update",
    ]):
        return "business_update"
    if any(x in text for x in [
        "brokerage", "citi research", "analyst", "target price",
    ]):
        return "analyst_research"

    return "unknown"


def source_quality(source_name: str) -> float:
    source = normalize_publisher_name(source_name)

    if source in {
        "reuters",
        "bloomberg",
        "official exchange",
        "company filing",
        "company investor relations",
    }:
        return 1.00

    if source in {
        "business standard",
        "moneycontrol",
        "economic times",
        "financial express",
        "livemint",
        "indian express",
    }:
        return 0.85

    return 0.60


def event_specificity(event_type: str) -> float:
    return {
        "earnings": 1.00,
        "guidance": 1.00,
        "block_trade": 0.95,
        "corporate_action": 0.95,
        "policy_event": 0.80,
        "business_update": 0.90,
        "analyst_research": 0.75,
        "sector_event": 0.65,
        "macro_event": 0.50,
        "unknown": 0.40,
    }.get(event_type, 0.40)


def temporal_relevance(event_date, anomaly_date) -> float:
    distance = abs((event_date - anomaly_date).days)

    if distance == 0:
        return 1.00
    if distance == 1:
        return 0.90
    if distance == 2:
        return 0.75
    if distance == 3:
        return 0.60
    if distance <= 7:
        return 0.30
    return 0.00


def direction_matches(event_direction: str, stock_return: float):
    if event_direction == "neutral":
        return None
    if stock_return > 0:
        return event_direction == "positive"
    if stock_return < 0:
        return event_direction == "negative"
    return None


def evidence_relevance_score(candidate: dict) -> float:
    direction = (
        1.00 if candidate["direction_matches"] is True
        else 0.00 if candidate["direction_matches"] is False
        else 0.50
    )

    return (
        candidate["temporal_relevance"] * 0.40
        + direction * 0.25
        + candidate["source_quality"] * 0.20
        + candidate["event_specificity"] * 0.15
    )


def build_evidence_candidate(
    symbol,
    anomaly_date,
    stock_return,
    event_date,
    event_type,
    headline,
    summary,
    source_name,
    source_url,
) -> dict:
    direction = classify_event_direction(
        f"{headline} {summary}"
    )

    candidate = {
        "symbol": symbol,
        "anomaly_date": anomaly_date,
        "event_date": event_date,
        "event_type": event_type,
        "headline": headline,
        "summary": summary,
        "source_name": source_name,
        "source_url": source_url,
        "event_direction": direction,
        "date_distance_days": abs(
            (event_date - anomaly_date).days
        ),
        "temporal_relevance": temporal_relevance(
            event_date,
            anomaly_date,
        ),
        "direction_matches": direction_matches(
            direction,
            stock_return,
        ),
    }

    candidate["source_quality"] = source_quality(source_name)
    candidate["event_specificity"] = event_specificity(event_type)
    candidate["evidence_relevance_score"] = evidence_relevance_score(
        candidate
    )

    return candidate


def build_search_evidence(target: dict, search_results: list[dict]) -> list[dict]:
    """Normalize provider results into scored evidence candidates."""
    candidates = []

    for result in search_results:
        event_date = result.get("published_date")
        if not event_date:
            continue

        if isinstance(event_date, str):
            from datetime import date
            event_date = date.fromisoformat(event_date)

        text = f"{result.get('title', '')} {result.get('text', '')}"

        event_type = result.get("event_type") or infer_event_type(text)

        candidate = build_evidence_candidate(
            symbol=target["symbol"],
            anomaly_date=target["anomaly_date"],
            stock_return=target["daily_return"],
            event_date=event_date,
            event_type=event_type,
            headline=result.get("title", ""),
            summary=result.get("text", ""),
            source_name=result.get("source_name")
            or source_domain(result.get("url")),
            source_url=result.get("url"),
        )

        candidates.append(candidate)

    return sorted(
        candidates,
        key=lambda x: x["evidence_relevance_score"],
        reverse=True,
    )


def consolidate_evidence(candidates: list[dict]) -> dict:
    """Deduplicate source families and summarize corroboration."""
    if not candidates:
        return {
            "candidate_count": 0,
            "independent_source_count": 0,
            "source_families": [],
            "direction_matches": 0,
            "direction_conflicts": 0,
            "best_candidate": None,
            "independent_candidates": [],
        }

    ranked = sorted(
        candidates,
        key=lambda x: x["evidence_relevance_score"],
        reverse=True,
    )

    independent = []
    seen_families = set()
    seen_headlines = set()

    for candidate in ranked:
        family = source_family(
            candidate["source_url"],
            candidate["source_name"],
        )
        headline = candidate["headline"].lower().strip()

        if headline in seen_headlines:
            continue
        if family in seen_families:
            continue

        seen_headlines.add(headline)
        seen_families.add(family)
        independent.append(candidate)

    return {
        "candidate_count": len(candidates),
        "independent_source_count": len(independent),
        "source_families": sorted(seen_families),
        "direction_matches": sum(
            c["direction_matches"] is True for c in independent
        ),
        "direction_conflicts": sum(
            c["direction_matches"] is False for c in independent
        ),
        "best_candidate": independent[0] if independent else None,
        "independent_candidates": independent,
    }


def calculate_evidence_strength(consolidated: dict) -> float:
    """Calculate support strength; this is not a causality probability."""
    best = consolidated["best_candidate"]

    if not best:
        return 0.0

    count = consolidated["independent_source_count"]

    corroboration = (
        1.00 if count >= 3
        else 0.80 if count == 2
        else 0.50 if count == 1
        else 0.00
    )

    matches = consolidated["direction_matches"]
    conflicts = consolidated["direction_conflicts"]
    total = matches + conflicts

    direction_agreement = (
        matches / total if total else 0.50
    )

    return min(
        best["evidence_relevance_score"] * 0.55
        + corroboration * 0.25
        + direction_agreement * 0.20,
        1.0,
    )
