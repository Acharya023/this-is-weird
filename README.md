# This Is Weird — Market Edition

The first version of **This Is Weird** discovers statistically unusual events in Indian market data, then investigates them using additional context.

The goal is not to predict prices or make trading recommendations. The system should discover events that are unusual enough to deserve investigation and explain why they may be interesting, with evidence separated from statistical detection.

## Current experiment status

The first Colab experiment successfully loaded a 2025 NSE slice from Hugging Face:

- **20 exploration symbols**
- **4,960 rows**
- **2025 trading history**
- Daily OHLCV-style market data

The experiment has now validated three useful anomaly signals:

1. **Magnitude anomaly** — a return is unusual relative to the stock's own history.
2. **Activity anomaly** — trading volume is unusually high relative to its recent baseline.
3. **Relationship anomaly** — a stock behaves substantially differently from another stock that normally moves with it.

A fourth layer will later investigate the strongest candidates using market context, company events, filings and external sources.

## First validated anomaly

A useful experimental example was **HCLTECH on 2025-01-14**:

- Daily return: **−8.84%**
- Relative volume: **4.77×** its 20-day median
- HCLTECH's movement diverged sharply from the historical INFY/HCLTECH relationship.
- NIFTY 50 was not experiencing a comparable broad-market decline that day.

The statistical detector therefore surfaced a concrete event that could then be investigated rather than simply ranking stocks by percentage change.

This example is an experiment, not a claim that the current detector is production-ready.

## Data architecture

- **GitHub:** source code, configuration and documentation.
- **SQLite:** bounded local/generated market history.
- **Historical bootstrap:** Hugging Face `tejhq/indian-markets`, whose NSE subset is built from official exchange bhavcopy data and refreshed daily.
- **Ongoing source:** NSE official reports.
- **Anomaly detector:** numerical/statistical detection first; investigation/explanation later.

We intentionally do **not** copy the full Hugging Face dataset or SQLite database into Git.

## First-stage scope

The initial bootstrap uses 20 exploration symbols. This keeps the experiment small enough to inspect while providing enough cross-stock variation to test anomaly detection.

The stored daily fields are:

- Symbol
- Series
- Trade date
- Previous close
- Open / High / Low / Close
- Last price
- Volume
- Turnover
- Number of trades

VWAP is left empty in the Hugging Face bootstrap because the published dataset schema used for the bootstrap does not expose VWAP. The later NSE ingestion path can populate it.

## Detection approach

The current experimental pipeline is:

```
Market data
    ↓
Daily return
    ↓
Stock-specific baseline
    ↓
Return anomaly / z-score
    ↓
Volume baseline
    ↓
Relative-volume anomaly
    ↓
Cross-stock relationship
    ↓
Divergence candidate
```

The important design principle is that the statistical layer should **find the anomaly first**. An LLM should not be responsible for deciding whether a numerical event is unusual.

The later investigation layer will take selected anomaly candidates and look for supporting or conflicting evidence.

## Efficient bootstrap

The Hugging Face NSE data is partitioned into Parquet data. The bootstrap downloads only the requested yearly data, filters the configured symbols locally with Polars, and writes only that bounded slice into SQLite.

This is much more efficient than copying the complete market dataset into the repository.

Install dependencies:

```bash
pip install -r requirements.txt
```

Then:

```bash
python src/ingestion/bootstrap_hf.py 5
```

The argument is the number of years to load.

## Next milestone

The next engineering milestone is to move from the 20-stock experiment toward a proper market-context engine:

1. Build a reliable NIFTY 50 universe.
2. Add NIFTY 50 index data.
3. Calculate market-relative returns.
4. Replace exploratory formulas with reusable anomaly-detection code.
5. Test anomaly candidates across a larger universe.
6. Add the investigation/evidence layer.
7. Produce the first automated **This Is Weird** discovery record.

The current Colab calculations are exploratory and should not yet be treated as the production detector.
