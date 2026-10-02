"""SQLite persistence for discoveries and evidence."""

import sqlite3
from pathlib import Path

DEFAULT_DB = Path("data/market.sqlite")
SCHEMA = Path("schema.sql")


def initialize(db_path: Path = DEFAULT_DB) -> None:
    """Create the application schema if it does not exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA.read_text())


def save_discovery(
    discovery: dict,
    db_path: Path = DEFAULT_DB,
) -> None:
    """Insert or replace one structured discovery."""
    initialize(db_path)

    columns = [
        "date", "symbol", "daily_return", "return_z",
        "volume_ratio", "market_divergence",
        "median_relative_return", "stocks_up", "stocks_down",
        "discovery_score", "peer_count", "peer_evidence_quality",
        "external_event", "explanation", "evidence_status",
        "confidence", "investigation_notes",
    ]

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"""
            INSERT OR REPLACE INTO discoveries
            ({", ".join(columns)})
            VALUES ({", ".join("?" for _ in columns)})
            """,
            tuple(discovery.get(column) for column in columns),
        )


def save_evidence(
    discovery_date,
    symbol: str,
    sources: list[dict],
    db_path: Path = DEFAULT_DB,
) -> None:
    """Replace stored evidence sources for one discovery."""
    initialize(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            DELETE FROM discovery_evidence
            WHERE discovery_date = ? AND symbol = ?
            """,
            (discovery_date, symbol),
        )

        conn.executemany(
            """
            INSERT INTO discovery_evidence (
                discovery_date, symbol, source_name, source_url,
                published_date, evidence_type, relevance,
                source_summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    discovery_date,
                    symbol,
                    source.get("source_name", "unknown"),
                    source.get("source_url"),
                    source.get("published_date"),
                    source.get("event_type"),
                    source.get("relevance"),
                    source.get("summary"),
                )
                for source in sources
            ],
        )
