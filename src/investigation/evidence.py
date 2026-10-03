"""Evidence normalization, scoring and event grouping."""

from urllib.parse import urlparse


DOMAIN_ALIASES = {
}

# Company names used to reject search results that only happen to contain a
# short stock symbol (for example, LT) or otherwise refer to another entity.
COMPANY_ALIASES = {
    "ASIANPAINT": ["asian paints", "asian paint"],
    "AXISBANK": ["axis bank"],
    "BAJFINANCE": ["bajaj finance"],
    "BHARTIARTL": ["bharti airtel", "airtel"],
    "HCLTECH": ["hcltech", "hcl tech"],
    "HDFCBANK": ["hdfc bank"],
    "HINDUNILVR": ["hindustan unilever", "hul"],
    "ICICIBANK": ["icici bank"],
    "INFY": ["infosys"],
    "KOTAKBANK": ["kotak mahindra bank", "kotak bank"],
    "LT": ["larsen & toubro", "larsen and toubro", "l&t"],
    "MARUTI": ["maruti suzuki", "maruti"],
    "NTPC": ["ntpc"],
    "ONGC": ["ongc", "oil and natural gas corporation"],
    "RELIANCE": ["reliance industries", "reliance"],
    "SBIN": ["state bank of india", "sbi"],
    "SUNPHARMA": ["sun pharma", "sun pharmaceutical"],
    "TCS": ["tata consultancy services", "tcs"],
    "TITAN": ["titan company", "titan"],
    "WIPRO": ["wipro"],
}


def entity_relevance(symbol: str, headline: str, summary: str) -> float:
    """Measure whether evidence actually refers to the discovered company."""
    text = f"{headline} {summary}".lower()
    aliases = COMPANY_ALIASES.get(symbol.upper(), [symbol.lower()])
    if any(alias in text for alias in aliases):
        return 1.0
    # Short symbols are especially prone to accidental matches. Do not give
    # them partial credit unless the symbol itself is reasonably distinctive.
    if len(symbol) >= 4 and symbol.lower() in text:
        return 1.0
    return 0.0

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


def source_family(url: str | None, source_name: str | None = None) -> str:
    """Collapse known syndication and publisher aliases into evidence families."""
    domain = source_domain(url)
    publisher = normalize_publisher_name(source_name)

    if domain == "news.google.com" and source_name:
        return publisher
    if publisher == "reuters" or domain in {"reuters.com", "reuters"}:
        return "reuters"
    return normalize_publisher_name(domain)


def classify_event_direction(text: str) -> str:
    text = text.lower()
    positive_terms = [
        "surge", "rally", "rose", "gains", "higher",
        "growth", "upgrade", "strong", "beats", "benefit",
        "boost", "boosts", "cut taxes", "positive",
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
    """Infer a likely catalyst from distinctive event language."""
    text = text.lower()
    patterns = {
        "block_trade": [
            ("block deal", 5), ("bulk deal", 5), ("stake sale", 5),
            ("stake sold", 5), ("sold stake", 5),
        ],
        "policy_event": [
            ("gst", 5), ("gst reform", 5), ("tax cut", 5),
            ("tax reform", 5), ("tax restructuring", 5),
            ("government policy", 5), ("policy announcement", 5),
            ("policy reform", 5), ("regulatory change", 4),
            ("government decision", 4),
        ],
        "corporate_action": [
            ("demerger", 5), ("rights issue", 5), ("buyback", 5),
            ("bonus issue", 5), ("stock split", 5), ("split", 3),
            ("dividend", 4),
        ],
        "guidance": [
            ("guidance", 5), ("forecast", 4), ("outlook", 4),
            ("raised its forecast", 5), ("cut its forecast", 5),
            ("lowered guidance", 5),
        ],
        "business_update": [
            ("business update", 5), ("sales update", 5),
            ("operational update", 5), ("q1 update", 5),
            ("q2 update", 5), ("q3 update", 5), ("q4 update", 5),
            ("sales rose", 2), ("sales grew", 2),
        ],
        "analyst_research": [
            ("analyst", 4), ("brokerage", 4), ("target price", 5),
            ("price target", 5), ("citi research", 5),
            ("research report", 4),
        ],
        "earnings": [
            ("earnings", 5), ("quarterly results", 5), ("quarterly result", 5),
            ("q1 results", 5), ("q2 results", 5), ("q3 results", 5),
            ("q4 results", 5), ("reported results", 5),
            ("results beat", 5), ("results missed", 5),
            ("earnings report", 5), ("net profit", 3),
            ("net interest income", 3), ("net interest margin", 3),
            ("nim", 2), ("provisions", 2), ("bad loans", 2),
        ],
    }
    scores = {
        event_type: sum(weight for phrase, weight in phrases if phrase in text)
        for event_type, phrases in patterns.items()
    }
    best_type, best_score = max(scores.items(), key=lambda item: item[1])
    return best_type if best_score >= 4 else "unknown"


def search_result_quality(article: dict, target_aliases: list[str]) -> float:
    """Score whether a search result looks like explanatory evidence."""
    title = str(article.get("title") or "").strip()
    text = str(article.get("text") or "").strip()
    combined = f"{title} {text}".lower()

    aliases = [alias.lower() for alias in target_aliases if alias.strip()]
    entity_match = 1.0 if any(alias in combined for alias in aliases) else 0.0
    if not entity_match:
        return 0.0

    score = 0.50

    explanatory_markers = (
        "why", "after", "due to", "because", "amid", "following",
        "results", "guidance", "outlook", "block deal", "bulk deal",
        "stake sale", "update", "missed", "beat", "cut forecast",
        "downgrade", "upgrade",
    )
    if any(marker in combined for marker in explanatory_markers):
        score += 0.30

    listicle_markers = (
        "stocks to watch", "top gainers", "top losers", "check full list",
        "among top gainers", "stocks in focus",
    )
    if any(marker in combined for marker in listicle_markers):
        score -= 0.35

    pure_price_markers = (
        "share price today", "stock price today", "live updates",
    )
    if any(marker in title.lower() for marker in pure_price_markers):
        score -= 0.15

    if text and len(text) >= 120:
        score += 0.10

    return max(0.0, min(score, 1.0))


def source_quality(source_name: str) -> float:
    source = normalize_publisher_name(source_name)
    if source in {
        "reuters", "bloomberg", "official exchange",
        "company filing", "company investor relations",
    }:
        return 1.00
    if source in {
        "business standard", "moneycontrol", "economic times",
        "financial express", "livemint", "indian express",
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
    # Entity relevance is a gate at the search-result boundary, not part of
    # the low-level candidate score. This keeps direct candidate construction
    # useful for grouping/testing while build_search_evidence still rejects
    # unrelated search hits (for example, LT vs. medical "local therapy").
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
    direction = classify_event_direction(f"{headline} {summary}")
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
        "date_distance_days": abs((event_date - anomaly_date).days),
        "temporal_relevance": temporal_relevance(event_date, anomaly_date),
        "direction_matches": direction_matches(direction, stock_return),
        "entity_relevance": entity_relevance(symbol, headline, summary),
    }
    candidate["source_quality"] = source_quality(source_name)
    candidate["event_specificity"] = event_specificity(event_type)
    candidate["evidence_relevance_score"] = evidence_relevance_score(candidate)
    candidate["source_family"] = source_family(source_url, source_name)
    return candidate


def build_search_evidence(target: dict, search_results: list[dict]) -> list[dict]:
    """Normalize provider results into scored, entity-relevant evidence."""
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
            source_name=(
                result.get("source_name")
                or source_domain(result.get("url"))
            ),
            source_url=result.get("url"),
        )

        # Search engines can return results matching a short ticker without
        # actually referring to the discovered company. Such results must not
        # enter the evidence pool.
        if candidate["entity_relevance"] > 0:
            candidates.append(candidate)

    return sorted(
        candidates,
        key=lambda x: x["evidence_relevance_score"],
        reverse=True,
    )


def consolidate_evidence(candidates: list[dict]) -> dict:
    """Deduplicate publishers/syndication and summarize raw evidence."""
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

    ranked = sorted(candidates, key=lambda x: x["evidence_relevance_score"], reverse=True)
    independent = []
    seen_families = set()
    seen_headlines = set()

    for candidate in ranked:
        family = candidate["source_family"]
        headline = candidate["headline"].lower().strip()
        if headline in seen_headlines or family in seen_families:
            continue
        seen_headlines.add(headline)
        seen_families.add(family)
        independent.append(candidate)

    return {
        "candidate_count": len(candidates),
        "independent_source_count": len(independent),
        "source_families": sorted(seen_families),
        "direction_matches": sum(c["direction_matches"] is True for c in independent),
        "direction_conflicts": sum(c["direction_matches"] is False for c in independent),
        "best_candidate": independent[0] if independent else None,
        "independent_candidates": independent,
    }


def build_event_groups(candidates: list[dict]) -> list[dict]:
    """Group evidence by inferred catalyst and score the groups.

    Publisher count is deliberately capped: several outlets may describe the
    same underlying report. The group score rewards temporal fit, direction
    agreement and a small amount of source-family diversity rather than raw
    article count.
    """
    groups = {}
    for candidate in candidates:
        event_type = candidate.get("event_type") or "unknown"
        groups.setdefault(event_type, []).append(candidate)

    event_groups = []
    for event_type, members in groups.items():
        ranked = sorted(
            members,
            key=lambda x: x["evidence_relevance_score"],
            reverse=True,
        )

        family_best = {}
        for candidate in ranked:
            family = candidate["source_family"]
            if family not in family_best:
                family_best[family] = candidate

        independent = sorted(
            family_best.values(),
            key=lambda x: x["evidence_relevance_score"],
            reverse=True,
        )
        top = independent[:3]

        matches = sum(c["direction_matches"] is True for c in independent)
        conflicts = sum(c["direction_matches"] is False for c in independent)
        directional = matches + conflicts
        direction_agreement = matches / directional if directional else 0.50

        family_diversity = min(len(independent) / 3.0, 1.0)
        best_score = independent[0]["evidence_relevance_score"] if independent else 0.0
        mean_top_score = (
            sum(c["evidence_relevance_score"] for c in top) / len(top)
            if top else 0.0
        )
        temporal_fit = (
            sum(c["temporal_relevance"] for c in top) / len(top)
            if top else 0.0
        )

        specificity = event_specificity(event_type)

        # Generic/unknown articles are useful fallback evidence, but a large
        # number of them must not outrank a smaller set of specific catalyst
        # evidence. Source-family diversity therefore gets less weight than
        # catalyst specificity.
        # Independent corroboration is useful evidence that a catalyst is
        # actually associated with the anomaly. Keep the contribution bounded
        # so article volume cannot overwhelm a strong specific catalyst.
        corroboration = min(len(independent) / 2.0, 1.0)

        group_score = min(
            0.35 * best_score
            + 0.20 * mean_top_score
            + 0.15 * temporal_fit
            + 0.15 * direction_agreement
            + 0.05 * family_diversity
            + 0.05 * corroboration
            + 0.05 * specificity,
            1.0,
        )

        event_groups.append({
            "event_type": event_type,
            "score": group_score,
            "candidate_count": len(members),
            "independent_source_count": len(independent),
            "source_families": [c["source_family"] for c in independent],
            "direction_matches": matches,
            "direction_conflicts": conflicts,
            "direction_agreement": direction_agreement,
            "event_specificity": specificity,
            "best_candidate": independent[0] if independent else None,
            "evidence_candidates": independent[:5],
        })

    return sorted(
        event_groups,
        key=lambda group: (
            group["score"],
            group["best_candidate"]["evidence_relevance_score"]
            if group["best_candidate"] else 0.0,
        ),
        reverse=True,
    )


def calculate_event_group_strength(group: dict | None) -> float:
    """Return catalyst support strength, not a causality probability."""
    return group["score"] if group else 0.0


def calculate_evidence_strength(consolidated: dict) -> float:
    """Backward-compatible raw-evidence support strength."""
    best = consolidated["best_candidate"]
    if not best:
        return 0.0

    count = consolidated["independent_source_count"]
    corroboration = 1.00 if count >= 3 else 0.80 if count == 2 else 0.50
    matches = consolidated["direction_matches"]
    conflicts = consolidated["direction_conflicts"]
    total = matches + conflicts
    direction_agreement = matches / total if total else 0.50

    return min(
        best["evidence_relevance_score"] * 0.55
        + corroboration * 0.25
        + direction_agreement * 0.20,
        1.0,
    )
