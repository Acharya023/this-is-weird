import unittest

import polars as pl

from src.detection.population_discovery import score_population_discoveries


class PopulationDiscoveryTests(unittest.TestCase):
    def test_population_discovery_does_not_require_market_or_fixed_symbols(self):
        rows = []
        for symbol, multiplier in [("AAA", 1.0), ("BBB", 1.0), ("CCC", 1.0)]:
            for day in range(1, 90):
                rows.append(
                    {
                        "symbol": symbol,
                        "date": f"2025-01-{day:02d}" if day <= 31 else (
                            f"2025-02-{day - 31:02d}" if day <= 59
                            else f"2025-03-{day - 59:02d}"
                        ),
                        "adj_close": 100.0 * multiplier + day * 0.1,
                        "volume": 1000,
                    }
                )

        frame = pl.DataFrame(rows)

        # Make one entity unusual on the final observation.
        frame = frame.with_columns(
            pl.when(
                (pl.col("symbol") == "CCC")
                & (pl.col("date") == "2025-03-31")
            )
            .then(pl.col("adj_close") * 1.20)
            .otherwise(pl.col("adj_close"))
            .alias("adj_close")
        )

        result = score_population_discoveries(frame)

        self.assertIn("population_discovery_score", result.columns)
        self.assertGreater(len(result), 0)
        self.assertNotIn("market_divergence", result.columns)


if __name__ == "__main__":
    unittest.main()
