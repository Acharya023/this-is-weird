"""Benchmark population-wide market discovery.

Run after restarting the runtime:

    git clone https://github.com/Acharya023/this-is-weird.git
    cd this-is-weird
    pip install -r requirements.txt
    python scripts/test_population_discovery.py

This intentionally does NOT use config/bootstrap_symbols.txt.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.population_discovery import score_population_discoveries
from src.ingestion.hf_market_history import load_adjusted_history


def main():
    history = load_adjusted_history(years=1)

    print(f"Rows loaded: {len(history):,}")
    print(f"Symbols discovered: {history.select('symbol').n_unique():,}")
    print(
        "Date range: "
        f"{history.select('date').min()} -> {history.select('date').max()}"
    )

    discoveries = score_population_discoveries(history)

    print("\nTop 25 population-wide candidates:")
    print(
        discoveries
        .head(25)
        .select([
            "date",
            "symbol",
            "daily_return",
            "return_z",
            "volume_ratio",
            "cross_return_score",
            "cross_volume_score",
            "population_discovery_score",
        ])
        .to_pandas()
        .to_string(index=False)
    )

    print("\nThis is a candidate generator.")
    print("No predefined symbol list or benchmark was used.")


if __name__ == "__main__":
    main()
