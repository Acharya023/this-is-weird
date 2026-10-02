CREATE TABLE IF NOT EXISTS market_daily (
    symbol TEXT NOT NULL,
    series TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    prev_close REAL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    last_price REAL,
    close REAL NOT NULL,
    vwap REAL,
    volume INTEGER,
    turnover REAL,
    trades INTEGER,
    PRIMARY KEY (symbol, series, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_market_date ON market_daily(trade_date);
CREATE INDEX IF NOT EXISTS idx_market_symbol_date ON market_daily(symbol, trade_date);
