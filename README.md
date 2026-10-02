# This Is Weird — Market Edition

The first version of **This Is Weird** focuses on discovering statistically unusual events in Indian market data.

## Architecture

- **GitHub:** source code, configuration and documentation only.
- **SQLite:** generated market history and derived data.
- **NSE:** primary source for historical daily equity data.
- **Anomaly detector:** will identify unusual moves without being told which stock/event to look for.

## First-stage data

Daily security-level data:

- Symbol
- Series
- Trade date
- Previous close
- Open / High / Low / Close
- VWAP
- Total traded quantity
- Turnover
- Number of trades

## Storage rule

Raw market data and the SQLite database are generated artifacts. They must **not** be committed to Git.

## Load an NSE CSV

```bash
python src/ingestion/nse_to_sqlite.py path/to/nse.csv
```

The next milestone is to load real NSE history and test whether the detector finds genuinely interesting anomalies rather than ordinary large moves.
