"""
Index constituent fetching module.

Provides constituent ticker lists for major global indices.
Uses a combination of:
1. Wikipedia scraping for well-documented indices
2. Hardcoded fallback lists for less accessible indices
3. ETF holdings as proxy when direct constituent data is unavailable
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def get_constituents(source: str) -> list[str]:
    """
    Get constituent tickers for an index.

    Args:
        source: Constituent source identifier from INDEX_CONFIG

    Returns:
        List of Yahoo Finance ticker symbols
    """
    fetcher_map = {
        "sp500": _get_sp500,
        "nasdaq100": _get_nasdaq100,
        "djia": _get_djia,
        "russell2000": _get_russell2000_proxy,
        "ftse100": _get_ftse100,
        "dax": _get_dax,
        "cac40": _get_cac40,
        "eurostoxx50": _get_eurostoxx50,
        "nikkei225": _get_nikkei225,
        "hangseng": _get_hangseng,
        "asx200": _get_asx200_proxy,
        "kospi200": _get_kospi200_proxy,
        "china_a50": _get_china_a50,
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


def _get_sp500() -> list[str]:
    """Fetch S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].tolist()
    # Fix tickers with dots (Yahoo Finance uses hyphens)
    tickers = [t.replace(".", "-") for t in tickers]
    return tickers


def _get_nasdaq100() -> list[str]:
    """Fetch NASDAQ 100 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    # Find the table with ticker column
    for table in tables:
        if "Ticker" in table.columns:
            return table["Ticker"].tolist()
    # Fallback: try common column names
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


def _get_russell2000_proxy() -> list[str]:
    """
    Russell 2000 has 2000 stocks - use IWM ETF top holdings as proxy.
    For breadth calculation, we sample representative stocks.
    """
    # Use a representative sample via the IWM ETF
    # In practice, you'd use a data provider API for full constituents
    # Here we return the ETF ticker for proxy-based breadth calculation
    return ["IWM"]  # Will use ETF-based breadth proxy


def _get_ftse100() -> list[str]:
    """Fetch FTSE 100 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
    tables = pd.read_html(url)
    for table in tables:
        for col in ["Ticker", "EPIC", "Symbol"]:
            if col in table.columns:
                tickers = table[col].tolist()
                # Add .L suffix for London Stock Exchange
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
    """Fetch Euro Stoxx 50 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/EURO_STOXX_50"
    tables = pd.read_html(url)
    for table in tables:
        if "Ticker" in table.columns:
            return table["Ticker"].tolist()
    # Fallback: use a known list of major Euro Stoxx 50 tickers
    return [
        "ABI.BR", "AD.AS", "AI.PA", "AIR.PA", "ALV.DE", "ASML.AS",
        "BAS.DE", "BAYN.DE", "BBVA.MC", "BMW.DE", "BNP.PA", "CRG.IR",
        "CS.PA", "DHL.DE", "DTE.DE", "ENEL.MI", "ENI.MI", "FLO.MI",
        "IBE.MC", "IFX.DE", "ISP.MI", "ITX.MC", "KER.PA", "LIN.DE",
        "MC.PA", "MBG.DE", "MRK.DE", "MUV2.DE", "OR.PA", "ORA.PA",
        "PHIA.AS", "RMS.PA", "SAF.PA", "SAN.PA", "SAN.MC", "SAP.DE",
        "SIE.DE", "SU.PA", "TTE.PA", "UCG.MI", "VOW3.DE",
    ]


def _get_nikkei225() -> list[str]:
    """Fetch Nikkei 225 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Nikkei_225"
    tables = pd.read_html(url)
    for table in tables:
        for col in ["Ticker", "Code", "Securities code"]:
            if col in table.columns:
                codes = table[col].tolist()
                # Convert to Yahoo Finance format (add .T for Tokyo)
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
                # Convert to Yahoo Finance format (add .HK, pad to 4 digits)
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


def _get_asx200_proxy() -> list[str]:
    """Use STW ETF as proxy for ASX 200 breadth."""
    # Full ASX 200 constituents are not easily scraped
    # Use a representative set of top ASX stocks
    return [
        "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX", "ANZ.AX",
        "MQG.AX", "WES.AX", "WOW.AX", "TLS.AX", "RIO.AX", "FMG.AX",
        "ALL.AX", "STO.AX", "WDS.AX", "COL.AX", "GMG.AX", "TCL.AX",
        "REA.AX", "XRO.AX", "SHL.AX", "QBE.AX", "ORG.AX", "IAG.AX",
        "ASX.AX", "AMP.AX", "CPU.AX", "BSL.AX", "JHX.AX", "BXB.AX",
        "NCM.AX", "MIN.AX", "S32.AX", "SUN.AX", "MPL.AX", "TWE.AX",
        "RHC.AX", "SEK.AX", "SOL.AX", "NHF.AX", "ILU.AX", "OZL.AX",
        "IGO.AX", "EVN.AX", "NST.AX", "WHC.AX", "APA.AX", "DXS.AX",
    ]


def _get_kospi200_proxy() -> list[str]:
    """Use representative Korean large-cap stocks for KOSPI 200 breadth."""
    return [
        "005930.KS", "000660.KS", "005935.KS", "051910.KS", "006400.KS",
        "035420.KS", "000270.KS", "005380.KS", "068270.KS", "035720.KS",
        "207940.KS", "012330.KS", "066570.KS", "003670.KS", "028260.KS",
        "055550.KS", "105560.KS", "034730.KS", "096770.KS", "017670.KS",
        "018260.KS", "032830.KS", "086790.KS", "011200.KS", "003550.KS",
        "009150.KS", "033780.KS", "030200.KS", "010130.KS", "036570.KS",
    ]


def _get_china_a50() -> list[str]:
    """Use representative China A50 large-cap stocks."""
    return [
        "601398.SS", "601288.SS", "601939.SS", "601328.SS", "600036.SS",
        "600519.SS", "601318.SS", "600276.SS", "000858.SZ", "600900.SS",
        "601166.SS", "600030.SS", "600887.SS", "601888.SS", "600809.SS",
        "000333.SZ", "002714.SZ", "600690.SS", "601012.SS", "600309.SS",
        "601899.SS", "002594.SZ", "600585.SS", "601668.SS", "600050.SS",
        "601088.SS", "600048.SS", "601601.SS", "000568.SZ", "601628.SS",
        "603259.SS", "002352.SZ", "601669.SS", "600000.SS", "601857.SS",
        "600104.SS", "002304.SZ", "603288.SS", "600016.SS", "601211.SS",
    ]


def _get_nifty50() -> list[str]:
    """Fetch Nifty 50 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/NIFTY_50"
    tables = pd.read_html(url)
    for table in tables:
        if "Symbol" in table.columns:
            tickers = table["Symbol"].tolist()
            return [f"{t}.NS" for t in tickers if isinstance(t, str)]
    raise ValueError("Could not find Nifty 50 constituent table")
