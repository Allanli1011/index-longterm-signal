# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest for a single index
python main.py --index SP500

# Run parameter optimization
python main.py --index SP500 --optimize

# Walk-forward optimization (more robust, slower)
python main.py --index SP500 --walk-forward

# Run all indices
python main.py --all --optimize --quick

# List available indices
python main.py --list

# Run unit tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_data_sources.py -v

# Run a specific test class
python -m pytest tests/test_data_sources.py::TestExtractAkshareCodes -v
```

## Architecture

The system runs a market-breadth strategy on index futures: it measures what % of constituent stocks are above their N-day moving average, then trades the index futures contract when that breadth hits extreme levels and reverses.

**Data flow**: `constituents.py` → tickers list → `price_data.py` → close prices DataFrame → `strategy/breadth.py` → breadth indicators → `strategy/signals.py` → signal column → `backtest/engine.py` → `BacktestResult`

### Key modules

**`config/indices.py`** — Central configuration for all 17 indices. Each entry declares the Yahoo Finance ticker (`index_ticker`), futures code (`futures_ticker`), how to fetch constituents (`constituent_source`), and optionally `price_source: "akshare"` for Chinese A-shares.

**`data/constituents.py`** — Returns a list of tickers for a given `constituent_source` string. Falls back through: AKShare → BaoStock → Wikipedia scraping → hardcoded list. Chinese indices return A-share codes in `sh.XXXXXX` / `sz.XXXXXX` format; all other indices return Yahoo Finance tickers.

**`data/price_data.py`** — Downloads and caches price data as parquet files in `.cache/price_data/`. Auto-detects whether to use `yfinance` or `akshare` based on ticker format. Key helpers: `_detect_source()`, `_normalize_ashare_code()`, `ashare_to_yfinance()`.

**`strategy/breadth.py`** — Stateless functions that compute breadth metrics on a prices DataFrame. The `BreadthAnalyzer` class wraps these and produces a signals DataFrame with columns: `breadth`, `momentum`, `divergence`, `multi_tf_breadth`.

**`strategy/signals.py`** — `StrategyParams` dataclass controls all thresholds and modes. `generate_signals()` adds a `signal` column (1=long, -1=short, 0=flat) to the breadth DataFrame. Four modes: `simple`, `momentum_confirmed`, `divergence`, `composite`.

**`backtest/engine.py`** — Event-driven backtester that processes the signal column day-by-day, applying stop-loss and trailing stop logic. Returns a `BacktestResult` dataclass with equity curve, trade list, and performance metrics.

**`backtest/optimizer.py`** — Grid search and walk-forward optimization over `DEFAULT_PARAM_GRID` / `FAST_PARAM_GRID`. Uses `joblib` for parallelism. Returns `OptimizationResult` with ranked parameter combinations.

**`analysis/visualization.py`** — Generates matplotlib reports and saves them to `output/`.

## Chinese A-share specifics

- Chinese indices (`CSI300`, `CSI500`, `CSI1000`, `CHINA_A50`) require `price_source: "akshare"` in their config entry.
- Constituent tickers use the `sh.XXXXXX` / `sz.XXXXXX` prefix format (not Yahoo's `.SS`/`.SZ` suffix format). Shanghai stocks start with `6`; Shenzhen with `0` or `3`.
- The `ashare_to_yfinance()` helper converts between the two formats when needed.
- `akshare` is the primary source; `baostock` is the fallback for CSI 300 and CSI 500. CSI 1000 has no baostock fallback.

## Adding a new index

1. Add an entry to `INDEX_CONFIG` in `config/indices.py`.
2. Add a fetcher function in `data/constituents.py` and register it in the `fetcher_map` inside `get_constituents()`.
3. For Chinese A-shares, add `"price_source": "akshare"` to the config entry.
4. Add a test for the new constituent source in `tests/test_data_sources.py`.
