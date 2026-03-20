"""
Price data fetching and caching module.

Supports two data pipelines:
1. Yahoo Finance (yfinance) — for international stocks and indices
2. AKShare / BaoStock — for Chinese A-share stocks and indices

All data is cached locally as parquet files to avoid redundant API calls.
"""

import logging
import hashlib
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "price_data"


# ============================================================================
# Public API — auto-routes to the right data source
# ============================================================================


def get_index_price(ticker: str, start: str = "2005-01-01",
                    end: str | None = None,
                    source: str = "auto") -> pd.DataFrame:
    """
    Fetch historical price data for an index.

    Args:
        ticker: Index ticker (e.g., "^GSPC", "000300.SS", "sh.000300")
        start: Start date
        end: End date (None = today)
        source: "auto", "yfinance", "akshare", or "baostock"
    """
    if source == "auto":
        source = _detect_source(ticker)

    if source == "akshare":
        return _get_ashare_index_price(ticker, start, end)
    elif source == "baostock":
        return _get_baostock_index_price(ticker, start, end)
    else:
        return _fetch_yfinance(ticker, start, end)


def get_bulk_close_prices(tickers: list[str], start: str = "2005-01-01",
                          end: str | None = None,
                          source: str = "auto") -> pd.DataFrame:
    """
    Fetch close prices for multiple tickers as a single DataFrame.

    Auto-detects whether to use yfinance or akshare based on ticker format.

    Returns:
        DataFrame with dates as index and tickers as columns
    """
    if not tickers:
        return pd.DataFrame()

    if source == "auto":
        source = _detect_source(tickers[0])

    if source in ("akshare", "baostock"):
        return _get_ashare_bulk_close(tickers, start, end, source)
    else:
        return _get_yfinance_bulk_close(tickers, start, end)


# ============================================================================
# Chinese A-share data via AKShare
# ============================================================================


def _get_ashare_index_price(ticker: str, start: str,
                             end: str | None) -> pd.DataFrame:
    """Fetch Chinese index price data via akshare."""
    cache_file = _get_cache_path(f"akshare_idx_{ticker}", start, end)
    cached = _load_cache(cache_file)
    if cached is not None:
        return cached

    try:
        import akshare as ak

        # Normalize ticker: "000300.SS" -> "000300", "sh.000300" -> "000300"
        code = _normalize_ashare_code(ticker)

        logger.info(f"Fetching A-share index {code} via akshare...")
        df = ak.stock_zh_index_daily(symbol=f"sh{code}")

        if df is None or df.empty:
            # Try Shenzhen
            df = ak.stock_zh_index_daily(symbol=f"sz{code}")

        if df is not None and not df.empty:
            df.index = pd.to_datetime(df["date"])
            df = df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })

            # Filter date range
            if start:
                df = df[df.index >= start]
            if end:
                df = df[df.index <= end]

            _save_cache(cache_file, df)
            return df

    except Exception as e:
        logger.warning(f"akshare index fetch failed for {ticker}: {e}")

    # Fallback to baostock
    return _get_baostock_index_price(ticker, start, end)


def _get_baostock_index_price(ticker: str, start: str,
                                end: str | None) -> pd.DataFrame:
    """Fetch Chinese index price data via baostock."""
    cache_file = _get_cache_path(f"baostock_idx_{ticker}", start, end)
    cached = _load_cache(cache_file)
    if cached is not None:
        return cached

    try:
        import baostock as bs

        code = _normalize_ashare_code(ticker)
        # baostock wants format like "sh.000300"
        if code.startswith("000") or code.startswith("399"):
            bs_code = f"sh.{code}" if not code.startswith("399") else f"sz.{code}"
        else:
            bs_code = f"sh.{code}"

        lg = bs.login()
        try:
            end_date = end or pd.Timestamp.now().strftime("%Y-%m-%d")
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume",
                start_date=start,
                end_date=end_date,
                frequency="d",
                adjustflag="2",  # 前复权
            )
            rows = []
            while (rs.error_code == '0') & rs.next():
                rows.append(rs.get_row_data())
            df = pd.DataFrame(rows, columns=rs.fields)
        finally:
            bs.logout()

        if df is not None and len(df) > 0:
            df.index = pd.to_datetime(df["date"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            _save_cache(cache_file, df)
            return df

    except Exception as e:
        logger.error(f"baostock index fetch failed for {ticker}: {e}")

    return pd.DataFrame()


def _get_ashare_bulk_close(tickers: list[str], start: str,
                            end: str | None,
                            source: str = "akshare") -> pd.DataFrame:
    """
    Fetch close prices for multiple A-share stocks.
    Uses akshare primary, baostock fallback.
    """
    cache_key = _bulk_cache_key(tickers, start, end, prefix="ashare")
    cached = _load_bulk_cache(cache_key)
    if cached is not None:
        return cached

    close_prices = pd.DataFrame()
    failed = []

    from tqdm import tqdm

    for ticker in tqdm(tickers, desc="Fetching A-share prices"):
        try:
            series = _get_single_ashare_close(ticker, start, end, source)
            if series is not None and len(series) > 0:
                close_prices[ticker] = series
            else:
                failed.append(ticker)
        except Exception as e:
            logger.debug(f"Failed to fetch {ticker}: {e}")
            failed.append(ticker)

    if failed:
        logger.warning(f"Failed {len(failed)}/{len(tickers)} tickers: "
                       f"{failed[:10]}{'...' if len(failed) > 10 else ''}")

    if not close_prices.empty:
        close_prices = close_prices.dropna(how="all")
        _save_bulk_cache(cache_key, close_prices)
        logger.info(f"Got A-share data: {len(close_prices.columns)} stocks, "
                     f"{len(close_prices)} trading days")

    return close_prices


def _get_single_ashare_close(ticker: str, start: str,
                               end: str | None,
                               source: str = "akshare") -> pd.Series | None:
    """Fetch close price for a single A-share stock."""
    # Try individual cache first
    cache_file = _get_cache_path(f"ashare_{ticker}", start, end)
    cached = _load_cache(cache_file)
    if cached is not None and "Close" in cached.columns:
        return cached["Close"]

    code = _normalize_ashare_code(ticker)

    # Try akshare
    if source in ("akshare", "auto"):
        try:
            import akshare as ak
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start.replace("-", ""),
                end_date=(end or pd.Timestamp.now().strftime("%Y%m%d")).replace("-", ""),
                adjust="qfq",  # 前复权
            )
            if df is not None and len(df) > 0:
                df.index = pd.to_datetime(df["日期"])
                result = df["收盘"].rename("Close")
                # Save to cache
                cache_df = pd.DataFrame({"Close": result})
                _save_cache(cache_file, cache_df)
                return result
        except Exception as e:
            logger.debug(f"akshare failed for {ticker}: {e}")

    # Fallback to baostock
    try:
        import baostock as bs
        # Determine baostock code
        if ticker.startswith(("sh.", "sz.")):
            bs_code = ticker
        elif code.startswith(("6", "9")):
            bs_code = f"sh.{code}"
        else:
            bs_code = f"sz.{code}"

        lg = bs.login()
        try:
            end_date = end or pd.Timestamp.now().strftime("%Y-%m-%d")
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,close",
                start_date=start,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )
            rows = []
            while (rs.error_code == '0') & rs.next():
                rows.append(rs.get_row_data())
        finally:
            bs.logout()

        if rows:
            df = pd.DataFrame(rows, columns=["date", "close"])
            df.index = pd.to_datetime(df["date"])
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            result = df["close"].rename("Close")
            cache_df = pd.DataFrame({"Close": result})
            _save_cache(cache_file, cache_df)
            return result
    except Exception as e:
        logger.debug(f"baostock failed for {ticker}: {e}")

    return None


# ============================================================================
# International data via Yahoo Finance
# ============================================================================


def _fetch_yfinance(ticker: str, start: str,
                     end: str | None) -> pd.DataFrame | None:
    """Fetch data via yfinance with caching."""
    cache_file = _get_cache_path(f"yf_{ticker}", start, end)
    cached = _load_cache(cache_file)
    if cached is not None:
        return cached

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, auto_adjust=True)

        if df is not None and len(df) > 0:
            _save_cache(cache_file, df)
            return df
    except Exception as e:
        logger.debug(f"yfinance error for {ticker}: {e}")

    return None


def _get_yfinance_bulk_close(tickers: list[str], start: str,
                               end: str | None) -> pd.DataFrame:
    """Bulk download close prices via yfinance."""
    cache_key = _bulk_cache_key(tickers, start, end, prefix="yf")
    cached = _load_bulk_cache(cache_key)
    if cached is not None:
        return cached

    try:
        import yfinance as yf
        logger.info(f"Bulk downloading {len(tickers)} tickers via yfinance...")
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
            close_prices = data[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            close_prices = pd.DataFrame()
            for ticker in tickers:
                try:
                    if ticker in data.columns.get_level_values(0):
                        close_prices[ticker] = data[ticker]["Close"]
                except (KeyError, TypeError):
                    logger.debug(f"No data for {ticker}")

        close_prices = close_prices.dropna(how="all")
        _save_bulk_cache(cache_key, close_prices)
        logger.info(f"Got data for {len(close_prices.columns)} tickers, "
                     f"{len(close_prices)} trading days")
        return close_prices

    except Exception as e:
        logger.error(f"yfinance bulk download failed: {e}")
        return pd.DataFrame()


# ============================================================================
# Helper functions
# ============================================================================


def _detect_source(ticker: str) -> str:
    """Auto-detect data source based on ticker format."""
    if isinstance(ticker, str):
        if ticker.startswith(("sh.", "sz.")):
            return "akshare"
        if ticker.endswith((".SS", ".SZ")):
            return "akshare"
        # Pure 6-digit Chinese stock code
        if len(ticker) == 6 and ticker.isdigit():
            return "akshare"
    return "yfinance"


def _normalize_ashare_code(ticker: str) -> str:
    """
    Normalize various A-share ticker formats to pure 6-digit code.
    "sh.600519" -> "600519"
    "600519.SS" -> "600519"
    "000300.SS" -> "000300"
    "sz.000858" -> "000858"
    """
    ticker = str(ticker).strip()
    if ticker.startswith(("sh.", "sz.")):
        return ticker[3:]
    if ticker.endswith((".SS", ".SZ")):
        return ticker[:-3]
    return ticker


# ============================================================================
# Cache management
# ============================================================================


def _get_cache_path(key: str, start: str, end: str | None) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_key = key.replace("/", "_").replace("=", "_").replace(".", "_")
    end_str = end or "latest"
    return CACHE_DIR / f"{safe_key}_{start}_{end_str}.parquet"


def _load_cache(cache_file: Path) -> pd.DataFrame | None:
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            if len(df) > 0:
                return df
        except Exception:
            pass
    return None


def _save_cache(cache_file: Path, df: pd.DataFrame) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file)


def _bulk_cache_key(tickers: list[str], start: str, end: str | None,
                     prefix: str = "") -> str:
    content = f"{prefix}_{sorted(tickers)}_{start}_{end}"
    return hashlib.md5(content.encode()).hexdigest()


def _load_bulk_cache(key: str) -> pd.DataFrame | None:
    cache_file = CACHE_DIR / f"bulk_{key}.parquet"
    return _load_cache(cache_file)


def _save_bulk_cache(key: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"bulk_{key}.parquet"
    df.to_parquet(cache_file)


def clear_cache():
    """Clear all cached price data."""
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        logger.info("Cache cleared")
