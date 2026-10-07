"""Population-wide anomaly discovery.

Unlike the original controlled-stock experiment, these functions operate on
whatever entities are present in the supplied universe.  The universe is not
part of the scoring logic.
"""

import polars as pl

from src.detection.anomaly_detector import add_market_features


def score_population_discoveries(
    frame: pl.DataFrame,
    window_volume: int = 20,
    window_return: int = 60,
) -> pl.DataFrame:
    """Rank unusual observations across the entire supplied population.

    The first version intentionally uses only signals that do not require a
    benchmark or predefined peer group:
      - historical return abnormality
      - historical volume abnormality
      - cross-sectional return extremeness
      - cross-sectional volume extremeness

    This is candidate generation, not final "weirdness" classification.
    """
    enriched = add_market_features(
        frame.select(["symbol", "date", "adj_close", "volume"]),
        window_volume=window_volume,
        window_return=window_return,
    )

    enriched = enriched.filter(
        pl.col("return_z").is_not_null()
        & pl.col("volume_ratio").is_not_null()
        & (pl.col("return_std") > 0)
        & (pl.col("volume_median") > 0)
    )

    return (
        enriched
        .with_columns([
            pl.col("daily_return").abs().alias("absolute_return"),
            pl.col("return_z").clip(0, 6).alias("return_z_capped"),
            pl.col("volume_ratio").log1p().clip(0, 6).alias("volume_log"),
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
            pl.len().over("date").alias("population_size"),
        ])
        .with_columns([
            (
                (pl.col("return_rank") - 1)
                / pl.when(pl.col("population_size") > 1)
                .then(pl.col("population_size") - 1)
                .otherwise(1)
            ).alias("cross_return_score"),
            (
                (pl.col("volume_rank") - 1)
                / pl.when(pl.col("population_size") > 1)
                .then(pl.col("population_size") - 1)
                .otherwise(1)
            ).alias("cross_volume_score"),
        ])
        .with_columns(
            (
                (pl.col("return_z_capped") / 6) * 0.55
                + (pl.col("volume_log") / 6) * 0.25
                + pl.col("cross_return_score") * 0.15
                + pl.col("cross_volume_score") * 0.05
            ).alias("population_discovery_score")
        )
        .sort("population_discovery_score", descending=True)
    )
