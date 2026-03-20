"""
Trading Signal Generation Module.

Generates long/short/flat signals based on breadth indicators.
Supports multiple signal generation modes:

1. Simple Threshold: Long when breadth < oversold, Short when > overbought
2. Momentum Confirmed: Threshold + breadth momentum confirmation
3. Divergence Enhanced: Threshold + divergence signals for stronger conviction
4. Composite: Weighted combination of all signals
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class StrategyParams:
    """Parameters controlling signal generation."""
    # Core breadth thresholds
    ma_period: int = 200
    breadth_oversold: float = 25.0
    breadth_overbought: float = 75.0
    exit_neutral_low: float = 40.0
    exit_neutral_high: float = 60.0

    # Momentum confirmation
    use_momentum: bool = True
    breadth_momentum_window: int = 10
    momentum_confirm_threshold: float = 2.0

    # Divergence signals
    use_divergence: bool = True
    divergence_lookback: int = 60
    divergence_boost: float = 5.0  # Relax thresholds by this much on divergence

    # Multi-timeframe
    use_multi_tf: bool = False
    multi_tf_weight: float = 0.3

    # Risk management
    stop_loss_pct: float = 5.0
    trailing_stop_pct: float = 3.0
    max_holding_days: int = 0  # 0 = no limit

    # Signal mode
    mode: str = "momentum_confirmed"  # simple, momentum_confirmed, divergence, composite

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyParams":
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# Signal constants
SIGNAL_LONG = 1
SIGNAL_SHORT = -1
SIGNAL_FLAT = 0


def generate_signals(signals_df: pd.DataFrame,
                     params: StrategyParams) -> pd.DataFrame:
    """
    Generate trading signals based on breadth indicators.

    Args:
        signals_df: DataFrame from BreadthAnalyzer.get_signals_df()
        params: Strategy parameters

    Returns:
        DataFrame with added columns: 'raw_signal', 'signal', 'signal_reason'
    """
    df = signals_df.copy()
    df["raw_signal"] = SIGNAL_FLAT
    df["signal"] = SIGNAL_FLAT
    df["signal_reason"] = ""

    mode_func = {
        "simple": _simple_threshold_signals,
        "momentum_confirmed": _momentum_confirmed_signals,
        "divergence": _divergence_enhanced_signals,
        "composite": _composite_signals,
    }

    func = mode_func.get(params.mode, _momentum_confirmed_signals)
    df = func(df, params)

    # Apply position management (holding logic with exits)
    df = _apply_position_management(df, params)

    return df


def _simple_threshold_signals(df: pd.DataFrame,
                               params: StrategyParams) -> pd.DataFrame:
    """
    Simple threshold-based signals.

    Long when breadth < oversold threshold
    Short when breadth > overbought threshold
    """
    breadth = df["breadth"]

    # Generate entry signals
    df.loc[breadth <= params.breadth_oversold, "raw_signal"] = SIGNAL_LONG
    df.loc[breadth >= params.breadth_overbought, "raw_signal"] = SIGNAL_SHORT

    df.loc[breadth <= params.breadth_oversold, "signal_reason"] = "breadth_oversold"
    df.loc[breadth >= params.breadth_overbought, "signal_reason"] = "breadth_overbought"

    return df


def _momentum_confirmed_signals(df: pd.DataFrame,
                                  params: StrategyParams) -> pd.DataFrame:
    """
    Threshold + momentum confirmation.

    For longs: breadth must be oversold AND momentum must be turning positive
    (breadth is recovering from lows). This avoids catching falling knives.

    For shorts: breadth must be overbought AND momentum turning negative
    (breadth starting to roll over from highs).
    """
    breadth = df["breadth"]
    momentum = df["momentum"]

    # Long: oversold + momentum turning up
    long_condition = (
        (breadth <= params.breadth_oversold) &
        (momentum >= params.momentum_confirm_threshold)
    )

    # Short: overbought + momentum turning down
    short_condition = (
        (breadth >= params.breadth_overbought) &
        (momentum <= -params.momentum_confirm_threshold)
    )

    df.loc[long_condition, "raw_signal"] = SIGNAL_LONG
    df.loc[short_condition, "raw_signal"] = SIGNAL_SHORT

    df.loc[long_condition, "signal_reason"] = "oversold_momentum_confirm"
    df.loc[short_condition, "signal_reason"] = "overbought_momentum_confirm"

    return df


def _divergence_enhanced_signals(df: pd.DataFrame,
                                   params: StrategyParams) -> pd.DataFrame:
    """
    Threshold signals enhanced by divergence detection.

    When divergence is present, thresholds are relaxed (don't need to be
    as extreme to trigger a signal), reflecting higher conviction.
    """
    breadth = df["breadth"]
    momentum = df["momentum"]
    bull_div = df.get("bullish_divergence", pd.Series(0, index=df.index))
    bear_div = df.get("bearish_divergence", pd.Series(0, index=df.index))

    boost = params.divergence_boost

    # Standard signals (momentum confirmed)
    long_std = (
        (breadth <= params.breadth_oversold) &
        (momentum >= params.momentum_confirm_threshold)
    )
    short_std = (
        (breadth >= params.breadth_overbought) &
        (momentum <= -params.momentum_confirm_threshold)
    )

    # Divergence-boosted signals (relaxed thresholds)
    long_div = (
        (bull_div == 1) &
        (breadth <= params.breadth_oversold + boost) &
        (momentum >= 0)  # Just need non-negative momentum
    )
    short_div = (
        (bear_div == 1) &
        (breadth >= params.breadth_overbought - boost) &
        (momentum <= 0)
    )

    df.loc[long_std, "raw_signal"] = SIGNAL_LONG
    df.loc[long_std, "signal_reason"] = "oversold_momentum"
    df.loc[long_div, "raw_signal"] = SIGNAL_LONG
    df.loc[long_div, "signal_reason"] = "bullish_divergence"

    df.loc[short_std, "raw_signal"] = SIGNAL_SHORT
    df.loc[short_std, "signal_reason"] = "overbought_momentum"
    df.loc[short_div, "raw_signal"] = SIGNAL_SHORT
    df.loc[short_div, "signal_reason"] = "bearish_divergence"

    return df


def _composite_signals(df: pd.DataFrame,
                        params: StrategyParams) -> pd.DataFrame:
    """
    Weighted composite of all signal types.

    Computes a signal score from -1 to +1 based on:
    - Breadth level score
    - Momentum score
    - Divergence score
    - Multi-timeframe agreement score

    Entry when composite score exceeds threshold.
    """
    breadth = df["breadth"]
    momentum = df["momentum"]

    # Breadth level score: linear scale from oversold to overbought
    mid = (params.breadth_oversold + params.breadth_overbought) / 2
    range_ = (params.breadth_overbought - params.breadth_oversold) / 2
    breadth_score = -(breadth - mid) / range_  # Inverted: low breadth = positive score
    breadth_score = breadth_score.clip(-1, 1)

    # Momentum score
    mom_score = momentum / (params.momentum_confirm_threshold * 5)
    mom_score = mom_score.clip(-1, 1)

    # Divergence score
    bull_div = df.get("bullish_divergence", pd.Series(0, index=df.index))
    bear_div = df.get("bearish_divergence", pd.Series(0, index=df.index))
    div_score = (bull_div.astype(float) - bear_div.astype(float)) * 0.5

    # Composite
    composite = (
        0.50 * breadth_score +
        0.30 * mom_score +
        0.20 * div_score
    )

    if params.use_multi_tf and "multi_tf_breadth" in df.columns:
        mtf = df["multi_tf_breadth"]
        mtf_score = -(mtf - mid) / range_
        mtf_score = mtf_score.clip(-1, 1)
        composite = (1 - params.multi_tf_weight) * composite + params.multi_tf_weight * mtf_score

    df["composite_score"] = composite

    # Threshold for entry: require significant conviction
    df.loc[composite >= 0.5, "raw_signal"] = SIGNAL_LONG
    df.loc[composite >= 0.5, "signal_reason"] = "composite_bullish"
    df.loc[composite <= -0.5, "raw_signal"] = SIGNAL_SHORT
    df.loc[composite <= -0.5, "signal_reason"] = "composite_bearish"

    return df


def _apply_position_management(df: pd.DataFrame,
                                params: StrategyParams) -> pd.DataFrame:
    """
    Convert raw signals into actual position signals with proper
    entry/exit logic.

    Rules:
    - Enter on raw signal
    - Exit long when breadth rises above exit_neutral_low
    - Exit short when breadth falls below exit_neutral_high
    - Honor stop loss and max holding period
    """
    position = SIGNAL_FLAT
    entry_price = 0.0
    entry_date_idx = 0
    max_price = 0.0
    min_price = float("inf")

    signals = []
    reasons = []

    for i in range(len(df)):
        raw = df["raw_signal"].iloc[i]
        breadth = df["breadth"].iloc[i]
        price = df["index_price"].iloc[i] if "index_price" in df.columns else 0

        # Check exit conditions for existing positions
        if position == SIGNAL_LONG:
            # Track for trailing stop
            if price > max_price:
                max_price = price

            # Exit conditions
            exit_reason = None

            if breadth >= params.exit_neutral_low:
                exit_reason = "breadth_neutral_exit"
            elif params.stop_loss_pct > 0 and entry_price > 0:
                if price <= entry_price * (1 - params.stop_loss_pct / 100):
                    exit_reason = "stop_loss"
            if exit_reason is None and params.trailing_stop_pct > 0 and max_price > 0:
                if price <= max_price * (1 - params.trailing_stop_pct / 100):
                    exit_reason = "trailing_stop"
            if (exit_reason is None and params.max_holding_days > 0 and
                    (i - entry_date_idx) >= params.max_holding_days):
                exit_reason = "max_holding"

            if exit_reason:
                position = SIGNAL_FLAT
                signals.append(SIGNAL_FLAT)
                reasons.append(exit_reason)
                continue

        elif position == SIGNAL_SHORT:
            if price < min_price:
                min_price = price

            exit_reason = None

            if breadth <= params.exit_neutral_high:
                exit_reason = "breadth_neutral_exit"
            elif params.stop_loss_pct > 0 and entry_price > 0:
                if price >= entry_price * (1 + params.stop_loss_pct / 100):
                    exit_reason = "stop_loss"
            if exit_reason is None and params.trailing_stop_pct > 0 and min_price > 0:
                if price >= min_price * (1 + params.trailing_stop_pct / 100):
                    exit_reason = "trailing_stop"
            if (exit_reason is None and params.max_holding_days > 0 and
                    (i - entry_date_idx) >= params.max_holding_days):
                exit_reason = "max_holding"

            if exit_reason:
                position = SIGNAL_FLAT
                signals.append(SIGNAL_FLAT)
                reasons.append(exit_reason)
                continue

        # Check entry conditions
        if position == SIGNAL_FLAT and raw != SIGNAL_FLAT:
            position = raw
            entry_price = price
            entry_date_idx = i
            max_price = price
            min_price = price
            signals.append(raw)
            reasons.append(df["signal_reason"].iloc[i])
        else:
            signals.append(position)
            reasons.append("hold" if position != SIGNAL_FLAT else "")

    df["signal"] = signals
    df["position_reason"] = reasons

    return df
