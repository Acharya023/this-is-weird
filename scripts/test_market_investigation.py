"""End-to-end reproducible market investigation test.

Run from the repository root:

    python scripts/test_market_investigation.py

This script intentionally performs the complete experiment from a fresh
runtime: download the bounded Hugging Face market slice, detect anomalies,
then investigate the top candidates through Google News RSS.

It is a test harness, not a production job.
"""

from pathlib import Path
import sys
from urllib.request import Request, urlopen

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.anomaly_detector import (
    add_market_context,
    add_market_features,
    score_discoveries,
    valid_anomaly_observations,
)
from src.investigation import google_news, runner


DATA_URL = (
    "https://huggingface.co/datasets/tejhq/indian-markets/"
    "resolve/main/prices_adjusted/nse_2025.parquet"
)

SYMBOL_FILE = ROOT / "config" / "bootstrap_symbols.txt"
CACHE_DIR = ROOT / ".test_cache"
PARQUET_PATH = CACHE_DIR / "nse_2025.parquet"


def download_data() -> Path:
    CACHE_DIR.mkdir(exist_ok=True)

    if PARQUET_PATH.exists():
        print(f"Using cached data: {PARQUET_PATH}")
        return PARQUET_PATH

    print("Downloading Hugging Face market data...")
    request = Request(
        DATA_URL,
        headers={"User-Agent": "ThisIsWeird/0.1 test"},
    )

    with urlopen(request, timeout=120) as response, PARQUET_PATH.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)

    return PARQUET_PATH


def load_symbols() -> list[str]:
    return [
        line.strip()
        for line in SYMBOL_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def main() -> None:
    print("=" * 90)
    print("THIS IS WEIRD — REPRODUCIBLE MARKET INVESTIGATION TEST")
    print("=" * 90)

    symbols = load_symbols()
    print(f"Configured symbols: {len(symbols)}")

    parquet = download_data()
    raw = pl.read_parquet(parquet)

    required = {"symbol", "date", "adj_close", "volume"}
    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(f"Missing expected columns: {sorted(missing)}")

    frame = (
        raw
        .filter(pl.col("symbol").is_in(symbols))
        .select(["symbol", "date", "adj_close", "volume"])
        .sort(["symbol", "date"])
    )

    print(f"Filtered rows: {frame.height}")
    print(f"Date range: {frame['date'].min()} -> {frame['date'].max()}")

    features = add_market_features(frame)

    market = (
        features
        .group_by("date")
        .agg(pl.col("daily_return").mean().alias("market_return"))
        .sort("date")
    )

    contextual = add_market_context(features, market)
    valid = valid_anomaly_observations(contextual)
    scored = score_discoveries(valid)

    top10 = scored.sort("discovery_score", descending=True).head(10)

    print("\n" + "=" * 90)
    print("TOP 10 ANOMALIES")
    print("=" * 90)

    print(
        top10.select([
            "date",
            "symbol",
            "daily_return",
            "return_z",
            "volume_ratio",
            "market_divergence",
            "discovery_score",
        ])
    )

    provider = google_news.GoogleNewsRSSProvider()

    print("\n" + "=" * 90)
    print("INVESTIGATION")
    print("=" * 90)

    for row in top10.to_dicts():
        result = runner.investigate_with_provider(
            row,
            provider,
            limit_per_query=5,
        )

        print(
            f"\n{result['symbol']} {result['date']}"
            f" | event={result['best_event_type']}"
            f" | strength={result['evidence_strength']:.3f}"
            f" | status={result['evidence_status']}"
        )

        print(
            f"  searched={result['search_candidates']}"
            f" | independent={result['independent_sources']}"
        )

        if result["best_headline"]:
            print(f"  best: {result['best_headline']}")

        for group in result["event_groups"][:5]:
            print(
                f"    - {group['event_type']}: "
                f"{group['score']:.3f}"
                f" | articles={group['candidate_count']}"
                f" | families={group['independent_source_count']}"
                f" | specificity={group['event_specificity']:.2f}"
            )

    print("\n" + "=" * 90)
    print("TEST COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
