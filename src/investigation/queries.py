"""Generate compact investigation queries for discovered anomalies."""

from datetime import date


def investigation_queries(symbol: str, event_date: date) -> list[str]:
    """Return broad queries; evidence logic performs date/relevance filtering."""
    date_text = event_date.strftime("%d %B %Y")
    month_text = event_date.strftime("%B %Y")

    return [
        f'"{symbol}" "{date_text}"',
        f'"{symbol}" "{month_text}" results OR earnings OR announcement',
    ]


def build_investigation_target(row: dict) -> dict:
    """Build a serializable investigation target from an anomaly row."""
    event_date = row["date"]

    return {
        "symbol": row["symbol"],
        "anomaly_date": event_date,
        "daily_return": row["daily_return"],
        "return_z": row["return_z"],
        "volume_ratio": row["volume_ratio"],
        "market_divergence": row["market_divergence"],
        "discovery_score": row["discovery_score"],
        "queries": investigation_queries(
            row["symbol"],
            event_date,
        ),
    }
