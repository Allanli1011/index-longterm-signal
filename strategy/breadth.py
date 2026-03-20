"""
Market Breadth Indicator Engine.

Calculates various breadth metrics based on the percentage of constituent
stocks trading above their moving averages.

Key indicators:
1. Breadth Ratio: % of stocks above N-day MA
2. Breadth Momentum: Rate of change of breadth ratio
3. Breadth Divergence: Divergence between price and breadth
4. Multi-timeframe Breadth: Composite of 50/100/200 day breadth
"""

import numpy as np
import pandas as pd


def calculate_breadth_ratio(close_prices: pd.DataFrame,
                            ma_period: int = 200) -> pd.Series:
    """
    Calculate the percentage of stocks trading above their N-day moving average.

    Args:
        close_prices: DataFrame with dates as index, tickers as columns
        ma_period: Moving average period (default 200)

    Returns:
        Series with dates as index, values are breadth ratio (0-100)
    """
    if close_prices.empty:
        return pd.Series(dtype=float)

    # Calculate moving average for each stock
    ma = close_prices.rolling(window=ma_period, min_periods=ma_period).mean()

    # Boolean mask: is each stock above its MA?
    above_ma = close_prices > ma

    # Count valid (non-NaN) stocks per day
    valid_count = close_prices.notna().sum(axis=1)

    # Count stocks above MA (NaN in above_ma counts as False)
    above_count = above_ma.sum(axis=1)

    # Calculate ratio as percentage
    breadth = (above_count / valid_count * 100).where(valid_count > 0)

    # Drop initial NaN period
    breadth = breadth.dropna()

    return breadth


def calculate_breadth_momentum(breadth: pd.Series,
                                window: int = 10) -> pd.Series:
    """
    Calculate the rate of change of the breadth ratio.

    Positive momentum = breadth is improving (recovery signal)
    Negative momentum = breadth is deteriorating (weakness signal)

    Args:
        breadth: Breadth ratio series
        window: Lookback window for momentum calculation

    Returns:
        Series of breadth momentum values (change in breadth over window)
    """
    return breadth.diff(window)


def calculate_breadth_acceleration(breadth: pd.Series,
                                   window: int = 10) -> pd.Series:
    """
    Second derivative of breadth - is the momentum itself accelerating?

    Useful for detecting turning points:
    - Breadth very low + momentum turning positive + acceleration positive
      = Strong reversal signal
    """
    momentum = calculate_breadth_momentum(breadth, window)
    return momentum.diff(window)


def calculate_multi_timeframe_breadth(close_prices: pd.DataFrame,
                                      periods: list[int] | None = None,
                                      weights: list[float] | None = None
                                      ) -> pd.Series:
    """
    Calculate composite breadth from multiple MA periods.

    Default: 50-day (20%), 100-day (30%), 200-day (50%)

    This gives a more nuanced view:
    - All three low = extreme oversold
    - 50-day recovering while 200-day still low = early recovery
    - All three high = extreme overbought
    """
    if periods is None:
        periods = [50, 100, 200]
    if weights is None:
        weights = [0.2, 0.3, 0.5]

    assert len(periods) == len(weights)
    assert abs(sum(weights) - 1.0) < 1e-6

    composite = None
    for period, weight in zip(periods, weights):
        b = calculate_breadth_ratio(close_prices, period)
        if composite is None:
            composite = b * weight
        else:
            # Align indices
            common_idx = composite.index.intersection(b.index)
            composite = composite.loc[common_idx] + b.loc[common_idx] * weight

    return composite


def detect_breadth_divergence(index_price: pd.Series,
                               breadth: pd.Series,
                               lookback: int = 60,
                               min_price_change: float = 0.0) -> pd.DataFrame:
    """
    Detect divergences between index price and breadth.

    Bullish divergence: Price makes lower low, breadth makes higher low
    Bearish divergence: Price makes higher high, breadth makes lower high

    Args:
        index_price: Series of index closing prices
        breadth: Breadth ratio series
        lookback: Window to find local extremes
        min_price_change: Minimum price change % to qualify as new extreme

    Returns:
        DataFrame with columns: ['bullish_divergence', 'bearish_divergence']
        Values are 1 (divergence detected) or 0
    """
    # Align series
    common_idx = index_price.index.intersection(breadth.index)
    price = index_price.loc[common_idx]
    brd = breadth.loc[common_idx]

    result = pd.DataFrame(index=common_idx)
    result["bullish_divergence"] = 0
    result["bearish_divergence"] = 0

    half = lookback // 2

    for i in range(lookback, len(price)):
        window_price = price.iloc[i - lookback:i]
        window_breadth = brd.iloc[i - lookback:i]

        # Find local lows (for bullish divergence)
        price_low_idx = window_price.idxmin()
        recent_price_low_idx = price.iloc[i - half:i].idxmin()

        if (price.loc[recent_price_low_idx] < price.loc[price_low_idx] * (1 + min_price_change / 100)):
            # Price made lower low - check if breadth made higher low
            breadth_at_price_low = brd.loc[price_low_idx]
            breadth_at_recent_low = brd.loc[recent_price_low_idx]

            if breadth_at_recent_low > breadth_at_price_low:
                result.iloc[i, result.columns.get_loc("bullish_divergence")] = 1

        # Find local highs (for bearish divergence)
        price_high_idx = window_price.idxmax()
        recent_price_high_idx = price.iloc[i - half:i].idxmax()

        if (price.loc[recent_price_high_idx] > price.loc[price_high_idx] * (1 - min_price_change / 100)):
            # Price made higher high - check if breadth made lower high
            breadth_at_price_high = brd.loc[price_high_idx]
            breadth_at_recent_high = brd.loc[recent_price_high_idx]

            if breadth_at_recent_high < breadth_at_price_high:
                result.iloc[i, result.columns.get_loc("bearish_divergence")] = 1

    return result


def calculate_breadth_percentile(breadth: pd.Series,
                                  lookback: int = 252) -> pd.Series:
    """
    Calculate the percentile rank of current breadth within its own history.

    Useful for adaptive thresholds: instead of fixed 25/75,
    use rolling percentile to adapt to market regimes.
    """
    def _percentile_rank(series):
        if len(series) < 2:
            return 50.0
        current = series.iloc[-1]
        return (series < current).sum() / (len(series) - 1) * 100

    return breadth.rolling(window=lookback, min_periods=lookback // 2).apply(
        _percentile_rank, raw=False
    )


class BreadthAnalyzer:
    """
    Complete breadth analysis pipeline for a single index.

    Computes all breadth indicators and stores them for strategy consumption.
    """

    def __init__(self, close_prices: pd.DataFrame, index_price: pd.Series,
                 ma_period: int = 200, momentum_window: int = 10,
                 divergence_lookback: int = 60):
        self.close_prices = close_prices
        self.index_price = index_price
        self.ma_period = ma_period
        self.momentum_window = momentum_window
        self.divergence_lookback = divergence_lookback

        self._compute_all()

    def _compute_all(self):
        """Compute all breadth indicators."""
        # Core breadth ratio
        self.breadth = calculate_breadth_ratio(
            self.close_prices, self.ma_period
        )

        # Multi-timeframe breadth
        self.multi_tf_breadth = calculate_multi_timeframe_breadth(
            self.close_prices
        )

        # Breadth momentum (rate of change)
        self.momentum = calculate_breadth_momentum(
            self.breadth, self.momentum_window
        )

        # Breadth acceleration
        self.acceleration = calculate_breadth_acceleration(
            self.breadth, self.momentum_window
        )

        # Breadth divergence
        self.divergence = detect_breadth_divergence(
            self.index_price, self.breadth, self.divergence_lookback
        )

        # Historical percentile
        self.percentile = calculate_breadth_percentile(self.breadth)

    def get_signals_df(self) -> pd.DataFrame:
        """
        Compile all indicators into a single DataFrame for strategy use.
        """
        # Align all series to common index
        common_idx = self.breadth.index
        for series in [self.momentum, self.multi_tf_breadth, self.percentile]:
            if series is not None and not series.empty:
                common_idx = common_idx.intersection(series.index)

        df = pd.DataFrame(index=common_idx)
        df["breadth"] = self.breadth.reindex(common_idx)
        df["multi_tf_breadth"] = self.multi_tf_breadth.reindex(common_idx)
        df["momentum"] = self.momentum.reindex(common_idx)
        df["acceleration"] = self.acceleration.reindex(common_idx)
        df["percentile"] = self.percentile.reindex(common_idx)

        if self.divergence is not None and not self.divergence.empty:
            div_reindexed = self.divergence.reindex(common_idx)
            df["bullish_divergence"] = div_reindexed["bullish_divergence"]
            df["bearish_divergence"] = div_reindexed["bearish_divergence"]
        else:
            df["bullish_divergence"] = 0
            df["bearish_divergence"] = 0

        df["index_price"] = self.index_price.reindex(common_idx)

        return df.dropna(subset=["breadth"])
