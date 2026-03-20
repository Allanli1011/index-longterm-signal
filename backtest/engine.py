"""
Backtesting Engine.

Simulates strategy execution on historical data and calculates
comprehensive performance metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class TradeRecord:
    """Record of a single completed trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_points: float
    holding_days: int
    entry_reason: str
    exit_reason: str


@dataclass
class BacktestResult:
    """Complete backtest results."""
    index_name: str
    params: dict
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    # Equity curve
    equity_curve: pd.Series = field(default_factory=pd.Series)
    benchmark_curve: pd.Series = field(default_factory=pd.Series)

    # Trade list
    trades: list[TradeRecord] = field(default_factory=list)

    # Summary statistics
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0
    benchmark_return_pct: float = 0.0
    benchmark_annual_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl_pct: float = 0.0
    avg_winner_pct: float = 0.0
    avg_loser_pct: float = 0.0
    max_consecutive_losses: int = 0
    total_trades: int = 0
    long_trades: int = 0
    short_trades: int = 0
    avg_holding_days: float = 0.0
    time_in_market_pct: float = 0.0
    max_drawdown_duration_days: int = 0

    # Signal data (for analysis)
    signals_df: pd.DataFrame = field(default_factory=pd.DataFrame)


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Supports:
    - Long and short positions
    - Transaction costs
    - Slippage modeling
    - Position sizing
    """

    def __init__(self, initial_capital: float = 100000.0,
                 commission_pct: float = 0.02,
                 slippage_pct: float = 0.01):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

    def run(self, signals_df: pd.DataFrame, index_name: str = "",
            params: dict | None = None) -> BacktestResult:
        """
        Run backtest on a signals DataFrame.

        Args:
            signals_df: DataFrame with columns: signal, index_price, breadth, etc.
            index_name: Name of the index being tested
            params: Strategy parameters used (for record-keeping)

        Returns:
            BacktestResult with full performance analysis
        """
        df = signals_df.copy()
        df = df.dropna(subset=["index_price", "signal"])

        if len(df) < 2:
            return BacktestResult(
                index_name=index_name,
                params=params or {},
                start_date=pd.Timestamp.now(),
                end_date=pd.Timestamp.now(),
            )

        # Calculate daily returns of the index
        df["index_return"] = df["index_price"].pct_change()

        # Strategy returns: signal * index return (with 1-day lag for execution)
        df["prev_signal"] = df["signal"].shift(1).fillna(0)
        df["strategy_return"] = df["prev_signal"] * df["index_return"]

        # Apply transaction costs on position changes
        df["position_change"] = df["prev_signal"].diff().abs()
        df["costs"] = df["position_change"] * (self.commission_pct + self.slippage_pct) / 100
        df["net_return"] = df["strategy_return"] - df["costs"]

        # Build equity curve
        df["equity"] = (1 + df["net_return"]).cumprod() * self.initial_capital
        df["benchmark"] = (1 + df["index_return"]).cumprod() * self.initial_capital

        # Extract trades
        trades = self._extract_trades(df)

        # Calculate metrics
        result = BacktestResult(
            index_name=index_name,
            params=params or {},
            start_date=df.index[0],
            end_date=df.index[-1],
            equity_curve=df["equity"],
            benchmark_curve=df["benchmark"],
            trades=trades,
            signals_df=df,
        )

        self._calculate_metrics(result, df)

        return result

    def _extract_trades(self, df: pd.DataFrame) -> list[TradeRecord]:
        """Extract individual trades from signal series."""
        trades = []
        in_trade = False
        entry_date = None
        entry_price = 0.0
        direction = 0
        entry_reason = ""

        for i in range(1, len(df)):
            current_signal = df["signal"].iloc[i]
            prev_signal = df["signal"].iloc[i - 1]

            # Position opened
            if not in_trade and current_signal != 0:
                in_trade = True
                entry_date = df.index[i]
                entry_price = df["index_price"].iloc[i]
                direction = current_signal
                entry_reason = df.get("position_reason", pd.Series("", index=df.index)).iloc[i]

            # Position closed
            elif in_trade and (current_signal == 0 or
                               (current_signal != 0 and current_signal != direction)):
                exit_date = df.index[i]
                exit_price = df["index_price"].iloc[i]

                if direction == 1:
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100

                pnl_points = (exit_price - entry_price) * direction
                holding_days = (exit_date - entry_date).days

                exit_reason = df.get("position_reason", pd.Series("", index=df.index)).iloc[i]

                trades.append(TradeRecord(
                    entry_date=entry_date,
                    exit_date=exit_date,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                    pnl_points=pnl_points,
                    holding_days=holding_days,
                    entry_reason=entry_reason,
                    exit_reason=exit_reason,
                ))

                # If flipped direction, open new trade
                if current_signal != 0 and current_signal != direction:
                    entry_date = df.index[i]
                    entry_price = df["index_price"].iloc[i]
                    direction = current_signal
                    entry_reason = df.get("position_reason", pd.Series("", index=df.index)).iloc[i]
                else:
                    in_trade = False

        # Close any open trade at end
        if in_trade:
            exit_date = df.index[-1]
            exit_price = df["index_price"].iloc[-1]
            if direction == 1:
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            pnl_points = (exit_price - entry_price) * direction

            trades.append(TradeRecord(
                entry_date=entry_date,
                exit_date=exit_date,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                pnl_points=pnl_points,
                holding_days=(exit_date - entry_date).days,
                entry_reason=entry_reason,
                exit_reason="end_of_data",
            ))

        return trades

    def _calculate_metrics(self, result: BacktestResult,
                           df: pd.DataFrame) -> None:
        """Calculate comprehensive performance metrics."""
        equity = df["equity"]
        benchmark = df["benchmark"]
        net_returns = df["net_return"].dropna()
        trading_days = len(df)
        years = trading_days / 252

        # Returns
        result.total_return_pct = (equity.iloc[-1] / self.initial_capital - 1) * 100
        result.annual_return_pct = (
            (equity.iloc[-1] / self.initial_capital) ** (1 / max(years, 0.01)) - 1
        ) * 100 if years > 0 else 0

        result.benchmark_return_pct = (benchmark.iloc[-1] / self.initial_capital - 1) * 100
        result.benchmark_annual_return_pct = (
            (benchmark.iloc[-1] / self.initial_capital) ** (1 / max(years, 0.01)) - 1
        ) * 100 if years > 0 else 0

        # Drawdown analysis
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max * 100
        result.max_drawdown_pct = drawdown.min()

        # Max drawdown duration
        underwater = drawdown < 0
        if underwater.any():
            groups = (~underwater).cumsum()
            underwater_periods = underwater.groupby(groups).sum()
            result.max_drawdown_duration_days = int(underwater_periods.max())

        # Risk-adjusted returns
        if net_returns.std() > 0:
            result.sharpe_ratio = (
                net_returns.mean() / net_returns.std() * np.sqrt(252)
            )
        else:
            result.sharpe_ratio = 0.0

        # Sortino (downside deviation)
        downside = net_returns[net_returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            result.sortino_ratio = (
                net_returns.mean() / downside.std() * np.sqrt(252)
            )

        # Calmar
        if result.max_drawdown_pct < 0:
            result.calmar_ratio = result.annual_return_pct / abs(result.max_drawdown_pct)

        # Trade statistics
        trades = result.trades
        result.total_trades = len(trades)

        if trades:
            pnls = [t.pnl_pct for t in trades]
            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p <= 0]

            result.long_trades = sum(1 for t in trades if t.direction == 1)
            result.short_trades = sum(1 for t in trades if t.direction == -1)
            result.win_rate = len(winners) / len(pnls) * 100 if pnls else 0
            result.avg_trade_pnl_pct = np.mean(pnls)
            result.avg_winner_pct = np.mean(winners) if winners else 0
            result.avg_loser_pct = np.mean(losers) if losers else 0
            result.avg_holding_days = np.mean([t.holding_days for t in trades])

            # Profit factor
            gross_profit = sum(winners) if winners else 0
            gross_loss = abs(sum(losers)) if losers else 0
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            # Max consecutive losses
            consecutive = 0
            max_consecutive = 0
            for p in pnls:
                if p <= 0:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            result.max_consecutive_losses = max_consecutive

        # Time in market
        in_market = (df["signal"] != 0).sum()
        result.time_in_market_pct = in_market / len(df) * 100

    def summary_dict(self, result: BacktestResult) -> dict:
        """Return a summary dictionary for easy comparison."""
        return {
            "Index": result.index_name,
            "Period": f"{result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}",
            "Total Return (%)": round(result.total_return_pct, 2),
            "Annual Return (%)": round(result.annual_return_pct, 2),
            "Benchmark Return (%)": round(result.benchmark_return_pct, 2),
            "Benchmark Annual (%)": round(result.benchmark_annual_return_pct, 2),
            "Max Drawdown (%)": round(result.max_drawdown_pct, 2),
            "Sharpe Ratio": round(result.sharpe_ratio, 2),
            "Sortino Ratio": round(result.sortino_ratio, 2),
            "Calmar Ratio": round(result.calmar_ratio, 2),
            "Win Rate (%)": round(result.win_rate, 2),
            "Profit Factor": round(result.profit_factor, 2),
            "Total Trades": result.total_trades,
            "Long Trades": result.long_trades,
            "Short Trades": result.short_trades,
            "Avg Trade PnL (%)": round(result.avg_trade_pnl_pct, 2),
            "Avg Winner (%)": round(result.avg_winner_pct, 2),
            "Avg Loser (%)": round(result.avg_loser_pct, 2),
            "Max Consec. Losses": result.max_consecutive_losses,
            "Avg Holding (days)": round(result.avg_holding_days, 1),
            "Time in Market (%)": round(result.time_in_market_pct, 1),
        }
