"""Market history and universe loading from Hugging Face.

The source dataset already contains a broad NSE history.  This module deliberately
does not require a hand-written symbol list: the observation universe is discovered
from the source data itself.

Hugging Face is the durable data layer; local files are only a runtime cache.
"""

from datetime import date
from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download

HF_DATASET = "tejhq/indian-markets"
HF_REPO_TYPE = "dataset"


def years_to_load(years: int) -> range:
    if years < 1:
        raise ValueError("years must be >= 1")
    end_year = date.today().year
    return range(end_year - years + 1, end_year + 1)


def download_adjusted_year(year: int) -> str:
    """Download one complete NSE adjusted-price year and return its local path."""
    return hf_hub_download(
        repo_id=HF_DATASET,
        filename=f"prices_adjusted/nse_{year}.parquet",
        repo_type=HF_REPO_TYPE,
    )


def discover_symbols(path: str) -> list[str]:
    """Discover the NSE EQ symbol universe present in a downloaded year."""
    return (
        pl.scan_parquet(path)
        .filter(pl.col("series") == "EQ")
        .select("symbol")
        .unique()
        .sort("symbol")
        .collect()
        .get_column("symbol")
        .to_list()
    )


def load_adjusted_history(
    years: int = 1,
    symbols: set[str] | None = None,
) -> pl.DataFrame:
    """Load adjusted NSE OHLCV history without requiring a fixed symbol list."""
    frames = []

    for year in years_to_load(years):
        path = download_adjusted_year(year)
        scan = (
            pl.scan_parquet(path)
            .filter(pl.col("series") == "EQ")
            .select([
                "date",
                "symbol",
                "isin",
                "name",
                "adj_close",
                "volume",
                "turnover",
            ])
        )

        if symbols:
            scan = scan.filter(pl.col("symbol").is_in(sorted(symbols)))

        frames.append(scan.collect())

    if not frames:
        return pl.DataFrame()

    return pl.concat(frames, how="vertical_relaxed").sort(["symbol", "date"])
