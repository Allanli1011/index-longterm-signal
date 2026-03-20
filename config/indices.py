"""
Global stock index futures configuration.

Each index entry contains:
- name: Human-readable name
- index_ticker: Yahoo Finance ticker for the spot index
- futures_ticker: Yahoo Finance ticker for the front-month futures
- constituent_source: How to fetch constituent tickers
- country: Country/region
- trading_hours: Approximate trading hours (UTC)
- tick_value: Dollar value per index point for the futures contract
- margin: Approximate initial margin requirement (USD)
"""
from __future__ import annotations

INDEX_CONFIG = {
    # ==================== US ====================
    "SP500": {
        "name": "S&P 500",
        "index_ticker": "^GSPC",
        "futures_ticker": "ES=F",
        "constituent_source": "sp500",
        "country": "US",
        "tick_value": 50.0,
        "margin": 15000,
        "description": "US large-cap benchmark, 500 stocks",
    },
    "NASDAQ100": {
        "name": "NASDAQ 100",
        "index_ticker": "^NDX",
        "futures_ticker": "NQ=F",
        "constituent_source": "nasdaq100",
        "country": "US",
        "tick_value": 20.0,
        "margin": 20000,
        "description": "US tech-heavy, 100 stocks",
    },
    "DJIA": {
        "name": "Dow Jones Industrial Average",
        "index_ticker": "^DJI",
        "futures_ticker": "YM=F",
        "constituent_source": "djia",
        "country": "US",
        "tick_value": 5.0,
        "margin": 10000,
        "description": "US blue-chip, 30 stocks",
    },
    "RUSSELL2000": {
        "name": "Russell 2000",
        "index_ticker": "^RUT",
        "futures_ticker": "RTY=F",
        "constituent_source": "russell2000",
        "country": "US",
        "tick_value": 50.0,
        "margin": 8000,
        "description": "US small-cap, 2000 stocks",
    },
    # ==================== Europe ====================
    "FTSE100": {
        "name": "FTSE 100",
        "index_ticker": "^FTSE",
        "futures_ticker": "Z=F",
        "constituent_source": "ftse100",
        "country": "UK",
        "tick_value": 10.0,
        "margin": 8000,
        "description": "UK large-cap, 100 stocks",
    },
    "DAX": {
        "name": "DAX 40",
        "index_ticker": "^GDAXI",
        "futures_ticker": "FDAX=F",
        "constituent_source": "dax",
        "country": "Germany",
        "tick_value": 25.0,
        "margin": 25000,
        "description": "German large-cap, 40 stocks",
    },
    "CAC40": {
        "name": "CAC 40",
        "index_ticker": "^FCHI",
        "futures_ticker": "FCE=F",
        "constituent_source": "cac40",
        "country": "France",
        "tick_value": 10.0,
        "margin": 6000,
        "description": "French large-cap, 40 stocks",
    },
    "EUROSTOXX50": {
        "name": "Euro Stoxx 50",
        "index_ticker": "^STOXX50E",
        "futures_ticker": "FESX=F",
        "constituent_source": "eurostoxx50",
        "country": "Europe",
        "tick_value": 10.0,
        "margin": 5000,
        "description": "Eurozone blue-chip, 50 stocks",
    },
    # ==================== Asia-Pacific ====================
    "NIKKEI225": {
        "name": "Nikkei 225",
        "index_ticker": "^N225",
        "futures_ticker": "NIY=F",
        "constituent_source": "nikkei225",
        "country": "Japan",
        "tick_value": 5.0,
        "margin": 10000,
        "description": "Japanese large-cap, 225 stocks",
    },
    "HANGSENG": {
        "name": "Hang Seng Index",
        "index_ticker": "^HSI",
        "futures_ticker": "HSI=F",
        "constituent_source": "hangseng",
        "country": "Hong Kong",
        "tick_value": 50.0,
        "margin": 12000,
        "description": "Hong Kong large-cap, ~80 stocks",
    },
    "ASX200": {
        "name": "S&P/ASX 200",
        "index_ticker": "^AXJO",
        "futures_ticker": "AP=F",
        "constituent_source": "asx200",
        "country": "Australia",
        "tick_value": 25.0,
        "margin": 8000,
        "description": "Australian large-cap, 200 stocks",
    },
    "KOSPI200": {
        "name": "KOSPI 200",
        "index_ticker": "^KS200",
        "futures_ticker": "KS200=F",
        "constituent_source": "kospi200",
        "country": "South Korea",
        "tick_value": 250000,  # KRW
        "margin": 15000000,  # KRW
        "description": "Korean large-cap, 200 stocks",
    },
    # ==================== China A-shares ====================
    "CSI300": {
        "name": "沪深300",
        "index_ticker": "000300.SS",
        "futures_ticker": "IF",
        "constituent_source": "csi300",
        "country": "China",
        "tick_value": 300.0,  # CNY
        "margin": 170000,  # CNY
        "price_source": "akshare",
        "description": "沪深两市大盘, 300只成份股",
    },
    "CSI500": {
        "name": "中证500",
        "index_ticker": "000905.SS",
        "futures_ticker": "IC",
        "constituent_source": "csi500",
        "country": "China",
        "tick_value": 200.0,  # CNY
        "margin": 130000,  # CNY
        "price_source": "akshare",
        "description": "沪深两市中盘, 500只成份股",
    },
    "CSI1000": {
        "name": "中证1000",
        "index_ticker": "000852.SS",
        "futures_ticker": "IM",
        "constituent_source": "csi1000",
        "country": "China",
        "tick_value": 200.0,  # CNY
        "margin": 100000,  # CNY
        "price_source": "akshare",
        "description": "沪深两市小盘, 1000只成份股",
    },
    "CHINA_A50": {
        "name": "FTSE China A50",
        "index_ticker": "^XIN9",
        "futures_ticker": "XIN=F",
        "constituent_source": "china_a50",
        "country": "China",
        "tick_value": 1.0,
        "margin": 2000,
        "price_source": "auto",
        "description": "China A-share large-cap, 50 stocks",
    },
    # ==================== Emerging ====================
    "NIFTY50": {
        "name": "Nifty 50",
        "index_ticker": "^NSEI",
        "futures_ticker": "NIFTY=F",
        "constituent_source": "nifty50",
        "country": "India",
        "tick_value": 1.0,
        "margin": 5000,
        "description": "Indian large-cap, 50 stocks",
    },
}

# Default strategy parameters (will be optimized per index)
DEFAULT_STRATEGY_PARAMS = {
    "ma_period": 200,           # Moving average period for breadth calculation
    "breadth_oversold": 25,     # % threshold for oversold (go long)
    "breadth_overbought": 75,   # % threshold for overbought (go short)
    "exit_neutral_low": 40,     # Exit long when breadth rises above this
    "exit_neutral_high": 60,    # Exit short when breadth falls below this
    "breadth_momentum_window": 10,  # Window for breadth rate of change
    "momentum_confirm_threshold": 2.0,  # Min breadth momentum for confirmation
    "use_divergence": True,     # Enable breadth divergence detection
    "divergence_lookback": 60,  # Days to look back for divergence
    "stop_loss_pct": 5.0,       # Stop loss percentage
    "trailing_stop_pct": 3.0,   # Trailing stop percentage
    "position_size_method": "fixed",  # 'fixed' or 'volatility_scaled'
}
