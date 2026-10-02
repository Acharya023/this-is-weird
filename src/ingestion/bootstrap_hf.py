"""Bootstrap a bounded NSE history from the Hugging Face indian-markets dataset.

This deliberately stores only the configured symbols and date window in SQLite.
The full Hugging Face dataset is never copied into the repository.
"""

import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

from datasets import load_dataset

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


def main(years: int = 5):
    end = date.today()
    start = end - timedelta(days=365 * years)
    wanted = symbols()

    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.executescript(SCHEMA.read_text())

    # Streaming prevents the full HF dataset from being materialized locally.
    ds = load_dataset(
        "tejhq/indian-markets",
        "nse",
        split="train",
        streaming=True,
    )

    inserted = 0
    scanned = 0

    for row in ds:
        scanned += 1
        symbol = str(row["symbol"]).upper()

        if symbol not in wanted:
            continue

        d = row["date"]
        if hasattr(d, "isoformat"):
            d = d.isoformat()
        else:
            d = str(d)

        if not (start.isoformat() <= d <= end.isoformat()):
            continue

        conn.execute(
            """INSERT OR REPLACE INTO market_daily
            (symbol, series, trade_date, prev_close, open, high, low,
             last_price, close, vwap, volume, turnover, trades)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                symbol,
                str(row.get("series") or "EQ"),
                d,
                row.get("prev_close"),
                row["open"],
                row["high"],
                row["low"],
                row.get("last"),
                row["close"],
                None,
                row.get("volume"),
                row.get("turnover"),
                row.get("trades"),
            ),
        )
        inserted += 1

        if inserted % 5000 == 0:
            conn.commit()
            print(f"Inserted {inserted:,} rows; scanned {scanned:,}")

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM market_daily").fetchone()[0]
    conn.close()

    print(
        f"Bootstrap complete: inserted {inserted:,} rows; "
        f"database total is {total:,} rows."
    )


if __name__ == "__main__":
    years = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(years)
