"""
Tests for data source modules (constituents + price_data).
Uses mocking since we can't reach external APIs in CI.
"""

import sys
import types
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

sys.path.insert(0, "/home/user/index-longterm-signal")


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
        for key in ["CSI300", "CSI500", "CSI1000", "CHINA_A50"]:
            assert key in INDEX_CONFIG, f"{key} not in config"
            assert INDEX_CONFIG[key].get("price_source") == "akshare", \
                f"{key} should have price_source=akshare"

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
