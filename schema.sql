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


CREATE TABLE IF NOT EXISTS discoveries (
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    daily_return REAL,
    return_z REAL,
    volume_ratio REAL,
    market_divergence REAL,
    median_relative_return REAL,
    stocks_up INTEGER,
    stocks_down INTEGER,
    discovery_score REAL,
    peer_count INTEGER,
    peer_evidence_quality TEXT,
    external_event TEXT,
    explanation TEXT,
    evidence_status TEXT,
    confidence TEXT,
    investigation_notes TEXT,
    PRIMARY KEY (date, symbol)
);

CREATE TABLE IF NOT EXISTS discovery_evidence (
    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    discovery_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_url TEXT,
    published_date TEXT,
    evidence_type TEXT,
    relevance TEXT,
    source_summary TEXT,
    FOREIGN KEY (discovery_date, symbol)
        REFERENCES discoveries(date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_discovery_evidence_event
ON discovery_evidence(discovery_date, symbol);
