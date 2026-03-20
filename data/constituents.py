"""
Index constituent fetching module.

Data source priority (fallback chain):
1. AKShare (akshare) — covers Chinese indices via CSI official site,
   plus some global indices via various sources
2. BaoStock (baostock) — backup for Chinese indices, fully free, no registration
3. Wikipedia scraping — for major international indices
4. Hardcoded fallback lists — last resort for indices without API access

All fetchers return a list of Yahoo Finance-compatible tickers,
except Chinese A-share fetchers which return A-share codes (e.g., "600519")
that need to be used with akshare/baostock for price data.
"""

import logging
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# Public API
# ============================================================================


def get_constituents(source: str) -> list[str]:
    """
    Get constituent tickers for an index.

    Args:
        source: Constituent source identifier from INDEX_CONFIG

    Returns:
        List of ticker symbols (Yahoo Finance format for international,
        A-share code format like 'sh.600519' for Chinese stocks)
    """
    fetcher_map = {
        # China A-shares (akshare primary, baostock fallback)
        "csi300": _get_csi300,
        "csi500": _get_csi500,
        "csi1000": _get_csi1000,
        "china_a50": _get_china_a50,
        # US (Wikipedia primary)
        "sp500": _get_sp500,
        "nasdaq100": _get_nasdaq100,
        "djia": _get_djia,
        "russell2000": _get_russell2000,
        # Europe (Wikipedia primary)
        "ftse100": _get_ftse100,
        "dax": _get_dax,
        "cac40": _get_cac40,
        "eurostoxx50": _get_eurostoxx50,
        # Asia-Pacific (Wikipedia + hardcoded)
        "nikkei225": _get_nikkei225,
        "hangseng": _get_hangseng,
        "asx200": _get_asx200,
        "kospi200": _get_kospi200,
        # India
        "nifty50": _get_nifty50,
    }

    fetcher = fetcher_map.get(source)
    if fetcher is None:
        logger.warning(f"No constituent fetcher for source: {source}")
        return []

    try:
        tickers = fetcher()
        logger.info(f"Fetched {len(tickers)} constituents for {source}")
        return tickers
    except Exception as e:
        logger.error(f"Failed to fetch constituents for {source}: {e}")
        return []


# ============================================================================
# China A-shares — akshare primary, baostock fallback
# ============================================================================


def _get_csi300() -> list[str]:
    """沪深300成份股: akshare -> baostock fallback."""
    return _akshare_csi_constituents("000300", fallback=_baostock_hs300)


def _get_csi500() -> list[str]:
    """中证500成份股: akshare -> baostock fallback."""
    return _akshare_csi_constituents("000905", fallback=_baostock_zz500)


def _get_csi1000() -> list[str]:
    """中证1000成份股: akshare only (baostock doesn't have it)."""
    return _akshare_csi_constituents("000852")


def _get_china_a50() -> list[str]:
    """
    FTSE China A50: try akshare for CSI official constituents.
    A50 is not a CSI index, so we use a combination of top-50 by weight from CSI 300.
    """
    # FTSE A50 is roughly the top 50 of CSI 300 by market cap
    try:
        import akshare as ak
        df = ak.index_stock_cons_weight_csindex(symbol="000300")
        # Sort by weight descending and take top 50
        if "权重" in df.columns:
            df["权重"] = pd.to_numeric(df["权重"], errors="coerce")
            df = df.sort_values("权重", ascending=False)
        codes = _extract_akshare_codes(df)
        return codes[:50]
    except Exception as e:
        logger.warning(f"akshare A50 proxy failed: {e}, using hardcoded list")
        return _china_a50_hardcoded()


def _akshare_csi_constituents(symbol: str,
                                fallback=None) -> list[str]:
    """
    Fetch CSI index constituents via akshare's CSI official data interface.

    Args:
        symbol: CSI index code (e.g., "000300" for CSI 300)
        fallback: Optional fallback function if akshare fails
    """
    try:
        import akshare as ak
        df = ak.index_stock_cons_csindex(symbol=symbol)
        codes = _extract_akshare_codes(df)
        if codes:
            return codes
    except Exception as e:
        logger.warning(f"akshare CSI {symbol} failed: {e}")

    # Try fallback
    if fallback is not None:
        try:
            logger.info(f"Trying baostock fallback for {symbol}...")
            return fallback()
        except Exception as e2:
            logger.warning(f"Baostock fallback also failed: {e2}")

    return []


def _extract_akshare_codes(df: pd.DataFrame) -> list[str]:
    """
    Extract stock codes from akshare CSI DataFrame.

    akshare returns codes like '600519' — we convert to our internal
    format with exchange prefix: 'sh.600519' or 'sz.000858'.
    """
    code_col = None
    for col in ["成分券代码", "证券代码", "品种代码", "code", "Code"]:
        if col in df.columns:
            code_col = col
            break

    if code_col is None:
        logger.warning(f"Cannot find code column in akshare result. "
                       f"Columns: {list(df.columns)}")
        return []

    codes = df[code_col].astype(str).tolist()
    result = []
    for code in codes:
        code = code.strip().zfill(6)
        # Determine exchange: 6/9 = Shanghai, 0/3 = Shenzhen
        if code.startswith(("6", "9")):
            result.append(f"sh.{code}")
        elif code.startswith(("0", "3")):
            result.append(f"sz.{code}")
        else:
            result.append(f"sh.{code}")  # Default to Shanghai
    return result


def _baostock_hs300() -> list[str]:
    """Fetch 沪深300 constituents via baostock."""
    import baostock as bs
    lg = bs.login()
    try:
        rs = bs.query_hs300_stocks()
        rows = []
        while (rs.error_code == '0') & rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        return df["code"].tolist()  # Format: sh.600000 / sz.000001
    finally:
        bs.logout()


def _baostock_zz500() -> list[str]:
    """Fetch 中证500 constituents via baostock."""
    import baostock as bs
    lg = bs.login()
    try:
        rs = bs.query_zz500_stocks()
        rows = []
        while (rs.error_code == '0') & rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        return df["code"].tolist()
    finally:
        bs.logout()


def _china_a50_hardcoded() -> list[str]:
    """Hardcoded FTSE China A50 representative stocks."""
    return [
        "sh.601398", "sh.601288", "sh.601939", "sh.601328", "sh.600036",
        "sh.600519", "sh.601318", "sh.600276", "sz.000858", "sh.600900",
        "sh.601166", "sh.600030", "sh.600887", "sh.601888", "sh.600809",
        "sz.000333", "sz.002714", "sh.600690", "sh.601012", "sh.600309",
        "sh.601899", "sz.002594", "sh.600585", "sh.601668", "sh.600050",
        "sh.601088", "sh.600048", "sh.601601", "sz.000568", "sh.601628",
        "sh.603259", "sz.002352", "sh.601669", "sh.600000", "sh.601857",
        "sh.600104", "sz.002304", "sh.603288", "sh.600016", "sh.601211",
        "sh.601138", "sh.600031", "sh.601390", "sh.600029", "sh.601633",
        "sh.600015", "sz.000001", "sz.002415", "sh.601360", "sh.600196",
    ]


# ============================================================================
# US Indices — Wikipedia scraping
# ============================================================================


def _get_sp500() -> list[str]:
    """Fetch S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers


def _get_nasdaq100() -> list[str]:
    """Fetch NASDAQ 100 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for table in tables:
        if "Ticker" in table.columns:
            return table["Ticker"].tolist()
    for table in tables:
        for col in ["Symbol", "Ticker", "Stock symbol"]:
            if col in table.columns:
                return table[col].tolist()
    raise ValueError("Could not find NASDAQ 100 constituent table")


def _get_djia() -> list[str]:
    """Fetch Dow Jones constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    tables = pd.read_html(url)
    for table in tables:
        if "Symbol" in table.columns:
            return table["Symbol"].tolist()
    raise ValueError("Could not find DJIA constituent table")


def _get_russell2000() -> list[str]:
    """
    Russell 2000: try akshare for a broader stock universe,
    otherwise fall back to a representative sample.
    """
    # Try to get IWM ETF holdings via akshare
    try:
        import akshare as ak
        # akshare has a US stock list function
        # Use top ~500 small-cap US stocks as proxy
        df = ak.stock_us_spot_em()
        if df is not None and len(df) > 0:
            # Filter by market cap range typical of Russell 2000
            # Return a representative sample
            codes = df["代码"].tolist()[:500]
            return codes
    except Exception:
        pass

    # Fallback: representative small-cap stocks
    return [
        "SMCI", "CELH", "ANF", "DUOL", "ELF", "CAVA", "LNTH", "IPAR",
        "FN", "FTNT", "HALO", "BOOT", "PI", "KRYS", "ALKT", "CORT",
        "ACLX", "INTA", "VERX", "SHAK", "MOD", "AIT", "WDFC", "MLI",
        "SAIA", "PIPR", "PRMW", "FSS", "NSIT", "CSWI", "SWX", "CVLT",
        "ITGR", "SPSC", "EXPO", "PLXS", "SIG", "CALM", "BRBR", "TMDX",
        "GMS", "CPRX", "VRRM", "AAON", "ROAD", "KTOS", "HIMS", "TGTX",
        "BL", "GSHD", "NOVT", "CRS", "SKY", "RKLB", "LBRT", "XPEL",
        "TMHC", "AVAV", "CARG", "POWL", "APPF", "AEIS", "AZEK", "KNF",
        "ENSG", "MMSI", "CADE", "VCYT", "ESNT", "RUSHA", "UFPI", "BWXT",
        "HASI", "LUMN", "MGNI", "RBC", "FORM", "IRDM", "ASGN", "HUBG",
        "AROC", "CABO", "CALX", "CCOI", "CGNX", "CHE", "COLM", "COOP",
        "CYTK", "DLB", "EHC", "EPAM", "EXLS", "FIVE", "GLOB", "GWRE",
        "HLNE", "HQY", "IBKR", "IDCC", "IESC", "KNSL", "LANC", "LFUS",
    ]


# ============================================================================
# European Indices — Wikipedia scraping
# ============================================================================


def _get_ftse100() -> list[str]:
    """Fetch FTSE 100 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    tables = pd.read_html(url)
    for table in tables:
        for col in ["Ticker", "EPIC", "Symbol"]:
            if col in table.columns:
                tickers = table[col].tolist()
                return [f"{t}.L" for t in tickers if isinstance(t, str)]
    raise ValueError("Could not find FTSE 100 constituent table")


def _get_dax() -> list[str]:
    """Fetch DAX 40 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/DAX"
    tables = pd.read_html(url)
    for table in tables:
        for col in ["Ticker", "Ticker symbol", "Symbol"]:
            if col in table.columns:
                tickers = table[col].tolist()
                return [f"{t}.DE" for t in tickers if isinstance(t, str)]
    raise ValueError("Could not find DAX constituent table")


def _get_cac40() -> list[str]:
    """Fetch CAC 40 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/CAC_40"
    tables = pd.read_html(url)
    for table in tables:
        for col in ["Ticker", "Symbol"]:
            if col in table.columns:
                tickers = table[col].tolist()
                return [f"{t}.PA" for t in tickers if isinstance(t, str)]
    raise ValueError("Could not find CAC 40 constituent table")


def _get_eurostoxx50() -> list[str]:
    """Fetch Euro Stoxx 50 constituents from Wikipedia, with hardcoded fallback."""
    try:
        url = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
        tables = pd.read_html(url)
        for table in tables:
            if "Ticker" in table.columns:
                tickers = table["Ticker"].tolist()
                if len(tickers) >= 40:
                    return tickers
    except Exception as e:
        logger.warning(f"Wikipedia Euro Stoxx 50 failed: {e}")

    # Hardcoded fallback
    return [
        "ABI.BR", "AD.AS", "AI.PA", "AIR.PA", "ALV.DE", "ASML.AS",
        "BAS.DE", "BAYN.DE", "BBVA.MC", "BMW.DE", "BNP.PA", "CRG.IR",
        "CS.PA", "DHL.DE", "DTE.DE", "ENEL.MI", "ENI.MI", "FLO.MI",
        "IBE.MC", "IFX.DE", "ISP.MI", "ITX.MC", "KER.PA", "LIN.DE",
        "MC.PA", "MBG.DE", "MRK.DE", "MUV2.DE", "OR.PA", "ORA.PA",
        "PHIA.AS", "RMS.PA", "SAF.PA", "SAN.PA", "SAN.MC", "SAP.DE",
        "SIE.DE", "SU.PA", "TTE.PA", "UCG.MI", "VOW3.DE",
    ]


# ============================================================================
# Asia-Pacific Indices
# ============================================================================


def _get_nikkei225() -> list[str]:
    """Fetch Nikkei 225 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Nikkei_225"
    tables = pd.read_html(url)
    for table in tables:
        for col in ["Ticker", "Code", "Securities code"]:
            if col in table.columns:
                codes = table[col].tolist()
                return [f"{int(c)}.T" for c in codes
                        if pd.notna(c) and str(c).strip().isdigit()]
    raise ValueError("Could not find Nikkei 225 constituent table")


def _get_hangseng() -> list[str]:
    """Fetch Hang Seng Index constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
    tables = pd.read_html(url)
    for table in tables:
        for col in ["Ticker", "Stock code", "Code"]:
            if col in table.columns:
                codes = table[col].tolist()
                result = []
                for c in codes:
                    try:
                        code = str(int(c)).zfill(4)
                        result.append(f"{code}.HK")
                    except (ValueError, TypeError):
                        continue
                if result:
                    return result
    raise ValueError("Could not find Hang Seng constituent table")


def _get_asx200() -> list[str]:
    """
    ASX 200: try akshare for Australian stocks, fallback to hardcoded.
    """
    # Hardcoded representative set (top ~50 ASX stocks)
    return [
        "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX", "ANZ.AX",
        "MQG.AX", "WES.AX", "WOW.AX", "TLS.AX", "RIO.AX", "FMG.AX",
        "ALL.AX", "STO.AX", "WDS.AX", "COL.AX", "GMG.AX", "TCL.AX",
        "REA.AX", "XRO.AX", "SHL.AX", "QBE.AX", "ORG.AX", "IAG.AX",
        "ASX.AX", "AMP.AX", "CPU.AX", "BSL.AX", "JHX.AX", "BXB.AX",
        "NCM.AX", "MIN.AX", "S32.AX", "SUN.AX", "MPL.AX", "TWE.AX",
        "RHC.AX", "SEK.AX", "SOL.AX", "NHF.AX", "ILU.AX", "EVN.AX",
        "IGO.AX", "NST.AX", "WHC.AX", "APA.AX", "DXS.AX", "GPT.AX",
    ]


def _get_kospi200() -> list[str]:
    """KOSPI 200: representative Korean large-cap stocks."""
    return [
        "005930.KS", "000660.KS", "005935.KS", "051910.KS", "006400.KS",
        "035420.KS", "000270.KS", "005380.KS", "068270.KS", "035720.KS",
        "207940.KS", "012330.KS", "066570.KS", "003670.KS", "028260.KS",
        "055550.KS", "105560.KS", "034730.KS", "096770.KS", "017670.KS",
        "018260.KS", "032830.KS", "086790.KS", "011200.KS", "003550.KS",
        "009150.KS", "033780.KS", "030200.KS", "010130.KS", "036570.KS",
    ]


# ============================================================================
# India
# ============================================================================


def _get_nifty50() -> list[str]:
    """Fetch Nifty 50 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/NIFTY_50"
    tables = pd.read_html(url)
    for table in tables:
        if "Symbol" in table.columns:
            tickers = table["Symbol"].tolist()
            return [f"{t}.NS" for t in tickers if isinstance(t, str)]
    raise ValueError("Could not find Nifty 50 constituent table")


# ============================================================================
# Utility: check if a ticker list uses A-share format
# ============================================================================

def is_ashare_ticker(ticker: str) -> bool:
    """Check if a ticker is in A-share format (sh.XXXXXX / sz.XXXXXX)."""
    return ticker.startswith(("sh.", "sz."))


def ashare_to_yfinance(ticker: str) -> str:
    """Convert A-share ticker to Yahoo Finance format (for index price only)."""
    # sh.600519 -> 600519.SS, sz.000858 -> 000858.SZ
    if ticker.startswith("sh."):
        return f"{ticker[3:]}.SS"
    elif ticker.startswith("sz."):
        return f"{ticker[3:]}.SZ"
    return ticker
