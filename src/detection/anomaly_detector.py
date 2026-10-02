"""Reusable market anomaly detection primitives.

Returns and rolling statistics should be calculated from adjusted prices.
Raw volume remains suitable for activity anomalies.
"""

import polars as pl


def add_market_features(
    frame: pl.DataFrame,
    window_volume: int = 20,
    window_return: int = 60,
) -> pl.DataFrame:
    """Add adjusted return, historical abnormality and volume features.

    Required columns:
        symbol, date, adj_close, volume

    The input is sorted by symbol/date before rolling calculations.
    """
    if window_volume < 2 or window_return < 2:
        raise ValueError("rolling windows must be >= 2")

    required = {"symbol", "date", "adj_close", "volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return (
        frame
        .sort(["symbol", "date"])
        .with_columns(
            (
                pl.col("adj_close")
                / pl.col("adj_close").shift(1).over("symbol")
                - 1
            ).alias("daily_return")
        )
        .with_columns([
            pl.col("volume")
            .rolling_median(window_size=window_volume)
            .over("symbol")
            .alias("volume_median"),
            pl.col("daily_return")
            .rolling_mean(window_size=window_return)
            .over("symbol")
            .alias("return_mean"),
            pl.col("daily_return")
            .rolling_std(window_size=window_return)
            .over("symbol")
            .alias("return_std"),
        ])
        .with_columns([
            (
                (
                    pl.col("daily_return") - pl.col("return_mean")
                )
                / pl.col("return_std")
            )
            .abs()
            .alias("return_z"),
            (
                pl.col("volume") / pl.col("volume_median")
            ).alias("volume_ratio"),
        ])
    )


def add_market_context(
    frame: pl.DataFrame,
    market: pl.DataFrame,
    market_return_column: str = "market_return",
) -> pl.DataFrame:
    """Join a market return series and calculate relative divergence."""
    required_frame = {"date", "daily_return"}
    required_market = {"date", market_return_column}

    missing_frame = required_frame - set(frame.columns)
    missing_market = required_market - set(market.columns)

    if missing_frame:
        raise ValueError(f"Missing frame columns: {sorted(missing_frame)}")
    if missing_market:
        raise ValueError(f"Missing market columns: {sorted(missing_market)}")

    return (
        frame
        .join(
            market.select(["date", market_return_column]),
            on="date",
            how="left",
        )
        .with_columns(
            (
                pl.col("daily_return") - pl.col(market_return_column)
            ).alias("market_divergence")
        )
    )


def valid_anomaly_observations(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep rows where the core anomaly signals are available."""
    required = {"return_z", "volume_ratio", "market_divergence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return frame.filter(
        pl.col("return_z").is_not_null()
        & pl.col("volume_ratio").is_not_null()
        & pl.col("market_divergence").is_not_null()
    )


def score_discoveries(frame: pl.DataFrame) -> pl.DataFrame:
    """Calculate the current discovery score.

    This is an experimental detector score, not a probability.
    """
    required = {
        "date",
        "daily_return",
        "return_z",
        "volume_ratio",
        "market_divergence",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return (
        frame
        .with_columns([
            pl.col("daily_return").abs().alias("absolute_return"),
            pl.col("market_divergence")
            .abs()
            .alias("absolute_market_divergence"),
            (
                pl.col("return_z").clip(0, 5) / 5
            ).alias("historical_return_score"),
            (
                pl.col("volume_ratio").log1p().clip(0, 5) / 5
            ).alias("historical_volume_score"),
        ])
        .with_columns([
            pl.col("absolute_return")
            .rank(method="average")
            .over("date")
            .alias("return_rank"),
            pl.col("volume_ratio")
            .rank(method="average")
            .over("date")
            .alias("volume_rank"),
            pl.len().over("date").alias("stocks_that_day"),
        ])
        .with_columns([
            (
                (pl.col("return_rank") - 1)
                / (pl.col("stocks_that_day") - 1)
            ).alias("cross_return_score"),
            (
                (pl.col("volume_rank") - 1)
                / (pl.col("stocks_that_day") - 1)
            ).alias("cross_volume_score"),
        ])
        .with_columns(
            (
                pl.col("historical_return_score") * 0.45
                + pl.col("historical_volume_score") * 0.25
                + pl.col("cross_return_score") * 0.15
                + pl.col("cross_volume_score") * 0.10
                + (
                    pl.col("absolute_market_divergence")
                    .clip(0, 0.10)
                    / 0.10
                ) * 0.05
            ).alias("discovery_score")
        )
    )
