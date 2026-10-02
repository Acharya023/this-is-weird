"""Load NSE security-wise historical CSV data into SQLite."""

import csv
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB = Path("data/market.sqlite")
SCHEMA = Path("schema.sql")


def num(value):
    if value is None:
        return None
    value = str(value).strip().replace(",", "")
    return None if value in {"", "-", "NA", "null"} else float(value)


def integer(value):
    x = num(value)
    return None if x is None else int(x)


def normalize_date(value):
    for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"Unsupported NSE date: {value!r}")


def load(csv_path):
    DB.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB)
    conn.executescript(SCHEMA.read_text())

    inserted = 0

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            symbol = row["Symbol"].strip()
            series = row.get("Series", "EQ").strip()
            trade_date = normalize_date(row["Date"])

            conn.execute(
                """INSERT OR REPLACE INTO market_daily
                (symbol, series, trade_date, prev_close, open, high, low,
                 last_price, close, vwap, volume, turnover, trades)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    symbol,
                    series,
                    trade_date,
                    num(row.get("Prev Close")),
                    num(row.get("Open Price")),
                    num(row.get("High Price")),
                    num(row.get("Low Price")),
                    num(row.get("Last Price")),
                    num(row.get("Close Price")),
                    num(row.get("VWAP")),
                    integer(row.get("Total Traded Quantity")),
                    num(row.get("Turnover")),
                    integer(row.get("No. of Trades")),
                ),
            )
            inserted += 1

    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM market_daily").fetchone()[0]
    conn.close()

    print(f"Loaded {inserted:,} rows; database now contains {count:,} rows: {DB}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python src/ingestion/nse_to_sqlite.py <nse.csv>"
        )

    load(sys.argv[1])
