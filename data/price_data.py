"""
Price data fetching and caching module.

Handles downloading historical price data from Yahoo Finance
with built-in caching to avoid redundant API calls.
"""

import logging
import os
import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "price_data"


def get_index_price(ticker: str, start: str = "2005-01-01",
                    end: str | None = None) -> pd.DataFrame:
    """Fetch historical price data for an index or futures contract."""
    return _fetch_cached(ticker, start, end)


def get_constituent_prices(tickers: list[str], start: str = "2005-01-01",
                           end: str | None = None,
                           show_progress: bool = True) -> dict[str, pd.DataFrame]:
    """
    Fetch historical price data for multiple constituent stocks.

    Returns:
        Dict mapping ticker -> DataFrame with OHLCV data
    """
    results = {}
    failed = []

    iterator = tqdm(tickers, desc="Fetching prices") if show_progress else tickers

    for ticker in iterator:
        try:
            df = _fetch_cached(ticker, start, end)
            if df is not None and len(df) > 0:
                results[ticker] = df
            else:
                failed.append(ticker)
        except Exception as e:
            logger.debug(f"Failed to fetch {ticker}: {e}")
            failed.append(ticker)

    if failed:
        logger.warning(f"Failed to fetch {len(failed)}/{len(tickers)} tickers: "
                       f"{failed[:10]}{'...' if len(failed) > 10 else ''}")

    return results


def get_bulk_close_prices(tickers: list[str], start: str = "2005-01-01",
                          end: str | None = None) -> pd.DataFrame:
    """
    Fetch close prices for multiple tickers and return as a single DataFrame.

    Returns:
        DataFrame with dates as index and tickers as columns
    """
    cache_key = _bulk_cache_key(tickers, start, end)
    cached = _load_bulk_cache(cache_key)
    if cached is not None:
        return cached

    logger.info(f"Bulk downloading {len(tickers)} tickers...")
    try:
        # Use yfinance bulk download for efficiency
        data = yf.download(
            tickers,
            start=start,
            end=end,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=True,
        )

        if len(tickers) == 1:
            # Single ticker returns different format
            close_prices = data[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            # Extract close prices from multi-level columns
            close_prices = pd.DataFrame()
            for ticker in tickers:
                try:
                    if ticker in data.columns.get_level_values(0):
                        close_prices[ticker] = data[ticker]["Close"]
                except (KeyError, TypeError):
                    logger.debug(f"No data for {ticker} in bulk download")

        close_prices = close_prices.dropna(how="all")
        _save_bulk_cache(cache_key, close_prices)
        logger.info(f"Got data for {len(close_prices.columns)} tickers, "
                     f"{len(close_prices)} trading days")
        return close_prices

    except Exception as e:
        logger.error(f"Bulk download failed: {e}")
        return pd.DataFrame()


def _fetch_cached(ticker: str, start: str, end: str | None) -> pd.DataFrame | None:
    """Fetch data with file-based caching."""
    cache_file = _get_cache_path(ticker, start, end)

    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            if len(df) > 0:
                return df
        except Exception:
            pass

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, auto_adjust=True)

        if df is not None and len(df) > 0:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_file)
            return df
    except Exception as e:
        logger.debug(f"Error fetching {ticker}: {e}")

    return None


def _get_cache_path(ticker: str, start: str, end: str | None) -> Path:
    """Generate cache file path for a ticker."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.replace("/", "_").replace("=", "_")
    end_str = end or "latest"
    return CACHE_DIR / f"{safe_ticker}_{start}_{end_str}.parquet"


def _bulk_cache_key(tickers: list[str], start: str, end: str | None) -> str:
    """Generate a cache key for bulk downloads."""
    content = f"{sorted(tickers)}_{start}_{end}"
    return hashlib.md5(content.encode()).hexdigest()


def _load_bulk_cache(key: str) -> pd.DataFrame | None:
    """Load bulk data from cache."""
    cache_file = CACHE_DIR / f"bulk_{key}.parquet"
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            return None
    return None


def _save_bulk_cache(key: str, df: pd.DataFrame) -> None:
    """Save bulk data to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"bulk_{key}.parquet"
    df.to_parquet(cache_file)


def clear_cache():
    """Clear all cached price data."""
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        logger.info("Cache cleared")
