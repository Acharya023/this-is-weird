"""Generate compact investigation queries for discovered anomalies."""

from datetime import date


COMPANY_QUERY_ALIASES = {
    "ASIANPAINT": "Asian Paints",
    "AXISBANK": "Axis Bank",
    "BAJFINANCE": "Bajaj Finance",
    "BHARTIARTL": "Bharti Airtel",
    "HCLTECH": "HCLTech",
    "HDFCBANK": "HDFC Bank",
    "HINDUNILVR": "Hindustan Unilever",
    "ICICIBANK": "ICICI Bank",
    "INFY": "Infosys",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen & Toubro",
    "MARUTI": "Maruti Suzuki",
    "NTPC": "NTPC",
    "ONGC": "ONGC",
    "RELIANCE": "Reliance Industries",
    "SBIN": "State Bank of India",
    "SUNPHARMA": "Sun Pharma",
    "TCS": "Tata Consultancy Services",
    "TITAN": "Titan Company",
    "WIPRO": "Wipro",
}


def investigation_queries(symbol: str, event_date: date) -> list[str]:
    """Return several broad historical queries.

    Search providers may rank current pages above historical event coverage,
    so we deliberately vary the query instead of relying on one exact phrase.
    The evidence layer performs the final date/relevance filtering.
    """
    date_text = event_date.strftime("%d %B %Y")
    month_text = event_date.strftime("%B %Y")
    year_text = event_date.strftime("%Y")

    company = COMPANY_QUERY_ALIASES.get(symbol.upper(), symbol)

    return [
        f"{company} {date_text}",
        f"{symbol} {date_text} why shares",
        f"{company} {month_text} {year_text} news",
        f"{symbol} {month_text} {year_text} event",
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
