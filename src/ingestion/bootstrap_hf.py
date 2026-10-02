"""Bootstrap a bounded NSE history from Hugging Face Parquet.

The source dataset is partitioned by exchange/year. We download only the
requested yearly files, filter to the configured symbols, and write the
bounded result into SQLite. The full HF dataset is never copied into the repo.
"""

import sqlite3
import sys
from datetime import date
from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "market.sqlite"
SCHEMA = ROOT / "schema.sql"
SYMBOLS_FILE = ROOT / "config" / "bootstrap_symbols.txt"


def symbols():
    return {
        x.strip().upper()
        for x in SYMBOLS_FILE.read_text().splitlines()
        if x.strip() and not x.startswith("#")
    }


def years_to_load(years: int):
    end_year = date.today().year
    return range(end_year - years + 1, end_year + 1)


def load_year(year: int, wanted: set[str], conn: sqlite3.Connection) -> int:
    path = hf_hub_download(
        repo_id="tejhq/indian-markets",
        filename=f"nse/year={year}/nse_{year}.parquet",
        repo_type="dataset",
    )

    df = (
        pl.scan_parquet(path)
        .filter(
            pl.col("symbol").is_in(sorted(wanted))
            & (pl.col("series") == "EQ")
        )
        .select(
            "symbol",
            "series",
            "date",
            "prev_close",
            "open",
            "high",
            "low",
            "last",
            "close",
            "volume",
            "turnover",
            "trades",
        )
        .collect()
    )

    rows = [
        (
            row["symbol"],
            row["series"],
            row["date"].isoformat(),
            row["prev_close"],
            row["open"],
            row["high"],
            row["low"],
            row["last"],
            row["close"],
            None,
            row["volume"],
            row["turnover"],
            row["trades"],
        )
        for row in df.iter_rows(named=True)
    ]

    conn.executemany(
        """INSERT OR REPLACE INTO market_daily
        (symbol, series, trade_date, prev_close, open, high, low,
         last_price, close, vwap, volume, turnover, trades)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()

    print(f"{year}: selected {len(rows):,} rows")
    return len(rows)


def main(years: int = 5):
    if years < 1:
        raise SystemExit("years must be >= 1")

    wanted = symbols()
    if not wanted:
        raise SystemExit("No symbols configured in config/bootstrap_symbols.txt")

    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.executescript(SCHEMA.read_text())

    total_inserted = 0

    try:
        for year in years_to_load(years):
            total_inserted += load_year(year, wanted, conn)

        total = conn.execute(
            "SELECT COUNT(*) FROM market_daily"
        ).fetchone()[0]
        symbols_loaded = conn.execute(
            "SELECT COUNT(DISTINCT symbol) FROM market_daily"
        ).fetchone()[0]
        date_range = conn.execute(
            "SELECT MIN(trade_date), MAX(trade_date) FROM market_daily"
        ).fetchone()
    finally:
        conn.close()

    print(
        f"Bootstrap complete: inserted {total_inserted:,} rows; "
        f"database contains {total:,} rows across {symbols_loaded} symbols; "
        f"date range {date_range[0]} to {date_range[1]}."
    )


if __name__ == "__main__":
    years = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(years)
