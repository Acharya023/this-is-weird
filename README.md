# This Is Weird — Market Edition

The first version of **This Is Weird** discovers statistically unusual events in Indian market data.

## Data architecture

- **GitHub:** source code, configuration and documentation.
- **SQLite:** bounded local/generated market history.
- **Historical bootstrap:** Hugging Face `tejhq/indian-markets`, whose NSE subset is built from official exchange bhavcopy data and refreshed daily.
- **Ongoing source:** NSE official reports.
- **Anomaly detector:** numerical/statistical detection first; investigation/explanation later.

We intentionally do **not** copy the full Hugging Face dataset or SQLite database into Git.

## First-stage scope

The bootstrap starts with 20 exploration symbols and a five-year window. This keeps the first experiment small enough to inspect while giving the detector enough history to establish normal behavior.

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

VWAP is left empty in the Hugging Face bootstrap because the published NSE schema does not expose VWAP. The later NSE ingestion path can populate it.

## Efficient bootstrap

The Hugging Face NSE data is partitioned into one Parquet rollup per year. The bootstrap therefore downloads only the requested yearly files, filters the configured symbols locally with Polars, and writes only that bounded slice into SQLite.

This is much more efficient than streaming all ~7.2M NSE rows and discarding almost all of them.

Install dependencies:

```bash
pip install -r requirements.txt
```

Then:

```bash
python src/ingestion/bootstrap_hf.py 5
```

The argument is the number of years to retain. The default is 5.

The next milestone is to calculate returns, rolling volatility and relative volume, then see whether the detector finds genuinely interesting anomalies.
