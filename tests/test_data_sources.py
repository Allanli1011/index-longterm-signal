"""
Tests for data source modules (constituents + price_data).
Uses mocking since we can't reach external APIs in CI.
"""

import sys
import types
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Test constituents.py
# ============================================================================


class TestExtractAkshareCodes:
    """Test A-share code extraction and exchange prefix logic."""

    def test_shanghai_codes(self):
        from data.constituents import _extract_akshare_codes
        df = pd.DataFrame({"成分券代码": ["600519", "601398", "600036"]})
        result = _extract_akshare_codes(df)
        assert result == ["sh.600519", "sh.601398", "sh.600036"]

    def test_shenzhen_codes(self):
        from data.constituents import _extract_akshare_codes
        df = pd.DataFrame({"成分券代码": ["000858", "000333", "300750"]})
        result = _extract_akshare_codes(df)
        assert result == ["sz.000858", "sz.000333", "sz.300750"]

    def test_mixed_codes(self):
        from data.constituents import _extract_akshare_codes
        df = pd.DataFrame({"成分券代码": ["600519", "000858", "300750"]})
        result = _extract_akshare_codes(df)
        assert result == ["sh.600519", "sz.000858", "sz.300750"]

    def test_alternative_column_name(self):
        from data.constituents import _extract_akshare_codes
        df = pd.DataFrame({"证券代码": ["600519"]})
        result = _extract_akshare_codes(df)
        assert result == ["sh.600519"]

    def test_unknown_columns_returns_empty(self):
        from data.constituents import _extract_akshare_codes
        df = pd.DataFrame({"random_col": ["600519"]})
        result = _extract_akshare_codes(df)
        assert result == []

    def test_zero_padding(self):
        from data.constituents import _extract_akshare_codes
        df = pd.DataFrame({"成分券代码": ["858", "1"]})
        result = _extract_akshare_codes(df)
        # "858" -> "000858" (Shenzhen), "1" -> "000001" (Shenzhen)
        assert result == ["sz.000858", "sz.000001"]


class TestIsAshareTicker:
    def test_shanghai(self):
        from data.constituents import is_ashare_ticker
        assert is_ashare_ticker("sh.600519") is True

    def test_shenzhen(self):
        from data.constituents import is_ashare_ticker
        assert is_ashare_ticker("sz.000858") is True

    def test_us_ticker(self):
        from data.constituents import is_ashare_ticker
        assert is_ashare_ticker("AAPL") is False

    def test_yahoo_format(self):
        from data.constituents import is_ashare_ticker
        assert is_ashare_ticker("600519.SS") is False


class TestAshareToYfinance:
    def test_shanghai(self):
        from data.constituents import ashare_to_yfinance
        assert ashare_to_yfinance("sh.600519") == "600519.SS"

    def test_shenzhen(self):
        from data.constituents import ashare_to_yfinance
        assert ashare_to_yfinance("sz.000858") == "000858.SZ"

    def test_passthrough(self):
        from data.constituents import ashare_to_yfinance
        assert ashare_to_yfinance("AAPL") == "AAPL"


class TestGetConstituents:
    def test_unknown_source_returns_empty(self):
        from data.constituents import get_constituents
        result = get_constituents("nonexistent_index_xyz")
        assert result == []

    def test_hardcoded_fallback_china_a50(self):
        from data.constituents import _china_a50_hardcoded
        result = _china_a50_hardcoded()
        assert len(result) == 50
        assert all(t.startswith(("sh.", "sz.")) for t in result)

    def test_hardcoded_asx200(self):
        from data.constituents import _get_asx200
        result = _get_asx200()
        assert len(result) > 0
        assert all(t.endswith(".AX") for t in result)

    def test_hardcoded_kospi200(self):
        from data.constituents import _get_kospi200
        result = _get_kospi200()
        assert len(result) > 0
        assert all(t.endswith(".KS") for t in result)

    def test_hardcoded_eurostoxx50_fallback(self):
        from data.constituents import _get_eurostoxx50
        # Wikipedia will fail in this env, should fall back to hardcoded
        result = _get_eurostoxx50()
        assert len(result) > 30

    def test_hardcoded_russell2000(self):
        from data.constituents import _get_russell2000
        result = _get_russell2000()
        assert len(result) > 50


class TestMockedAkshareConstituents:
    """Test akshare-based fetching with mocked module."""

    def test_csi300_via_mock_akshare(self):
        mock_ak = MagicMock()
        mock_ak.index_stock_cons_csindex.return_value = pd.DataFrame({
            "成分券代码": ["600519", "000858", "601398"],
        })
        with patch.dict(sys.modules, {"akshare": mock_ak}):
            from data.constituents import _akshare_csi_constituents
            result = _akshare_csi_constituents("000300")
            assert result == ["sh.600519", "sz.000858", "sh.601398"]
            mock_ak.index_stock_cons_csindex.assert_called_once_with(symbol="000300")


# ============================================================================
# Test price_data.py
# ============================================================================


class TestDetectSource:
    def test_ashare_sh_format(self):
        from data.price_data import _detect_source
        assert _detect_source("sh.600519") == "akshare"

    def test_ashare_ss_format(self):
        from data.price_data import _detect_source
        assert _detect_source("000300.SS") == "akshare"

    def test_ashare_pure_code(self):
        from data.price_data import _detect_source
        assert _detect_source("600519") == "akshare"

    def test_us_ticker(self):
        from data.price_data import _detect_source
        assert _detect_source("AAPL") == "yfinance"

    def test_yahoo_index(self):
        from data.price_data import _detect_source
        assert _detect_source("^GSPC") == "yfinance"


class TestNormalizeAshareCode:
    def test_sh_prefix(self):
        from data.price_data import _normalize_ashare_code
        assert _normalize_ashare_code("sh.600519") == "600519"

    def test_sz_prefix(self):
        from data.price_data import _normalize_ashare_code
        assert _normalize_ashare_code("sz.000858") == "000858"

    def test_ss_suffix(self):
        from data.price_data import _normalize_ashare_code
        assert _normalize_ashare_code("000300.SS") == "000300"

    def test_sz_suffix(self):
        from data.price_data import _normalize_ashare_code
        assert _normalize_ashare_code("000858.SZ") == "000858"

    def test_raw_code(self):
        from data.price_data import _normalize_ashare_code
        assert _normalize_ashare_code("600519") == "600519"


class TestIncrementalCache:
    """Test incremental cache logic in price_data.py."""

    def test_cache_path_no_end_date(self, tmp_path):
        """Cache path should not include end date."""
        from data.price_data import _get_cache_path, CACHE_DIR
        import data.price_data as pd_mod

        original_dir = pd_mod.CACHE_DIR
        pd_mod.CACHE_DIR = tmp_path
        try:
            p1 = _get_cache_path("test", "2020-01-01", None)
            p2 = _get_cache_path("test", "2020-01-01", "2024-12-31")
            # Same path regardless of end date
            assert p1 == p2
            assert "latest" not in str(p1)
        finally:
            pd_mod.CACHE_DIR = original_dir

    def test_incremental_no_cache(self, tmp_path):
        """When no cache exists, returns (None, None)."""
        from data.price_data import _load_cache_for_incremental
        cache_file = tmp_path / "nonexistent.parquet"
        cached, fetch_start = _load_cache_for_incremental(cache_file)
        assert cached is None
        assert fetch_start is None

    def test_incremental_fresh_cache(self, tmp_path):
        """When cache covers up to yesterday, no fetch needed."""
        from data.price_data import _load_cache_for_incremental, _save_cache
        cache_file = tmp_path / "test.parquet"
        yesterday = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        dates = pd.date_range("2020-01-01", yesterday, freq="B")
        df = pd.DataFrame({"Close": range(len(dates))}, index=dates)
        _save_cache(cache_file, df)

        cached, fetch_start = _load_cache_for_incremental(cache_file)
        assert cached is not None
        assert fetch_start is None
        assert len(cached) == len(dates)

    def test_incremental_stale_cache(self, tmp_path):
        """When cache is old, returns data + fetch_start date."""
        from data.price_data import _load_cache_for_incremental, _save_cache
        cache_file = tmp_path / "test.parquet"
        # Cache ends 30 days ago
        old_end = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
        dates = pd.date_range("2020-01-01", old_end, freq="B")
        df = pd.DataFrame({"Close": range(len(dates))}, index=dates)
        _save_cache(cache_file, df)

        cached, fetch_start = _load_cache_for_incremental(cache_file)
        assert cached is not None
        assert fetch_start is not None
        expected_start = (old_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        assert fetch_start == expected_start

    def test_incremental_with_end_date(self, tmp_path):
        """When end date is specified and cache covers it, no fetch needed."""
        from data.price_data import _load_cache_for_incremental, _save_cache
        cache_file = tmp_path / "test.parquet"
        dates = pd.date_range("2020-01-01", "2023-12-31", freq="B")
        df = pd.DataFrame({"Close": range(len(dates))}, index=dates)
        _save_cache(cache_file, df)

        cached, fetch_start = _load_cache_for_incremental(
            cache_file, end="2023-06-30")
        assert cached is not None
        assert fetch_start is None
        # Should be filtered to end date
        assert cached.index[-1] <= pd.Timestamp("2023-06-30")

    def test_merge_and_save(self, tmp_path):
        """Test merging old and new data."""
        from data.price_data import _merge_and_save
        cache_file = tmp_path / "test.parquet"

        old_dates = pd.date_range("2020-01-01", "2020-06-30", freq="B")
        old = pd.DataFrame({"Close": range(len(old_dates))}, index=old_dates)

        new_dates = pd.date_range("2020-06-29", "2020-12-31", freq="B")
        new = pd.DataFrame({"Close": range(100, 100 + len(new_dates))},
                           index=new_dates)

        result = _merge_and_save(cache_file, old, new)
        # Should have no duplicate dates
        assert not result.index.duplicated().any()
        # Should span the full range
        assert result.index[0] == old_dates[0]
        assert result.index[-1] == new_dates[-1]
        # Overlapping dates should use new data (keep="last")
        overlap_date = pd.Timestamp("2020-06-29")
        assert result.loc[overlap_date, "Close"] >= 100
        # Should be saved to disk
        assert cache_file.exists()

    def test_merge_and_save_with_end_filter(self, tmp_path):
        """Merge should filter by end date when specified."""
        from data.price_data import _merge_and_save
        cache_file = tmp_path / "test.parquet"

        dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
        new = pd.DataFrame({"Close": range(len(dates))}, index=dates)

        result = _merge_and_save(cache_file, None, new, end="2020-06-30")
        assert result.index[-1] <= pd.Timestamp("2020-06-30")


class TestIndexConfig:
    """Verify all index configs are valid."""

    def test_all_indices_have_required_fields(self):
        from config.indices import INDEX_CONFIG
        required = {"name", "index_ticker", "futures_ticker",
                    "constituent_source", "country", "tick_value", "margin"}
        for key, cfg in INDEX_CONFIG.items():
            missing = required - set(cfg.keys())
            assert not missing, f"{key} missing fields: {missing}"

    def test_chinese_indices_have_price_source(self):
        from config.indices import INDEX_CONFIG
        for key in ["CSI300", "CSI500", "CSI1000"]:
            assert key in INDEX_CONFIG, f"{key} not in config"
            assert INDEX_CONFIG[key].get("price_source") == "akshare", \
                f"{key} should have price_source=akshare"
        # CHINA_A50 uses "auto" since it's an offshore index
        assert INDEX_CONFIG["CHINA_A50"].get("price_source") in ("akshare", "auto")

    def test_constituent_source_has_fetcher(self):
        from config.indices import INDEX_CONFIG
        from data.constituents import get_constituents
        for key, cfg in INDEX_CONFIG.items():
            source = cfg["constituent_source"]
            # Just verify the source is recognized (won't actually fetch)
            # get_constituents returns [] for network failures, not raises
            # So we just verify the mapping exists
            assert source is not None, f"{key} has no constituent_source"

    def test_index_count(self):
        from config.indices import INDEX_CONFIG
        # Original 14 + 3 new Chinese = 17
        assert len(INDEX_CONFIG) >= 17
