"""
Analysis and Visualization Module.

Generates charts and reports for backtest results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

from backtest.engine import BacktestResult, BacktestEngine

# Style configuration
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "equity": "#2196F3",
    "benchmark": "#9E9E9E",
    "long": "#4CAF50",
    "short": "#F44336",
    "breadth": "#FF9800",
    "drawdown": "#E91E63",
    "neutral": "#607D8B",
}

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def plot_backtest_report(result: BacktestResult, save: bool = True,
                         show: bool = False) -> str | None:
    """
    Generate a comprehensive multi-panel backtest report.

    Panels:
    1. Equity curve vs benchmark
    2. Drawdown chart
    3. Breadth indicator with signal markers
    4. Monthly returns heatmap
    5. Trade distribution
    """
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.25)

    index_name = result.index_name or "Unknown Index"
    fig.suptitle(f"Breadth Strategy Backtest Report: {index_name}",
                 fontsize=18, fontweight="bold", y=0.98)

    # --- Panel 1: Equity Curve ---
    ax1 = fig.add_subplot(gs[0, :])
    _plot_equity_curve(ax1, result)

    # --- Panel 2: Drawdown ---
    ax2 = fig.add_subplot(gs[1, :])
    _plot_drawdown(ax2, result)

    # --- Panel 3: Breadth with Signals ---
    ax3 = fig.add_subplot(gs[2, :])
    _plot_breadth_signals(ax3, result)

    # --- Panel 4: Monthly Returns Heatmap ---
    ax4 = fig.add_subplot(gs[3, 0])
    _plot_monthly_heatmap(ax4, result)

    # --- Panel 5: Trade Distribution ---
    ax5 = fig.add_subplot(gs[3, 1])
    _plot_trade_distribution(ax5, result)

    # --- Panel 6: Summary Stats ---
    ax6 = fig.add_subplot(gs[4, :])
    _plot_summary_table(ax6, result)

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = index_name.replace(" ", "_").replace("/", "_")
        filepath = OUTPUT_DIR / f"backtest_{safe_name}.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        return str(filepath)

    if show:
        plt.show()
    plt.close(fig)
    return None


def _plot_equity_curve(ax, result: BacktestResult):
    """Plot equity curve with benchmark comparison."""
    equity = result.equity_curve
    benchmark = result.benchmark_curve

    if equity.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    ax.plot(equity.index, equity.values, color=COLORS["equity"],
            linewidth=1.5, label="Strategy", zorder=3)
    ax.plot(benchmark.index, benchmark.values, color=COLORS["benchmark"],
            linewidth=1.0, alpha=0.7, label="Buy & Hold", zorder=2)

    # Shade long/short periods
    df = result.signals_df
    if "signal" in df.columns:
        long_mask = df["signal"] == 1
        short_mask = df["signal"] == -1

        for mask, color, label in [
            (long_mask, COLORS["long"], "Long"),
            (short_mask, COLORS["short"], "Short")
        ]:
            if mask.any():
                ax.fill_between(equity.index, equity.min() * 0.95,
                                equity.max() * 1.05,
                                where=mask.reindex(equity.index, fill_value=False),
                                alpha=0.08, color=color, label=label)

    ax.set_title("Equity Curve", fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)


def _plot_drawdown(ax, result: BacktestResult):
    """Plot drawdown chart."""
    equity = result.equity_curve
    if equity.empty:
        return

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max * 100

    ax.fill_between(drawdown.index, drawdown.values, 0,
                    color=COLORS["drawdown"], alpha=0.4)
    ax.plot(drawdown.index, drawdown.values, color=COLORS["drawdown"],
            linewidth=0.8)

    ax.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)

    # Annotate max drawdown
    min_dd_idx = drawdown.idxmin()
    min_dd = drawdown.min()
    ax.annotate(f"Max DD: {min_dd:.1f}%",
                xy=(min_dd_idx, min_dd),
                fontsize=10, color=COLORS["drawdown"], fontweight="bold")


def _plot_breadth_signals(ax, result: BacktestResult):
    """Plot breadth indicator with buy/sell signal markers."""
    df = result.signals_df
    if df.empty or "breadth" not in df.columns:
        return

    ax.plot(df.index, df["breadth"], color=COLORS["breadth"],
            linewidth=1.0, label="Breadth (% > 200 DMA)")

    # Threshold lines
    params = result.params
    oversold = params.get("breadth_oversold", 25)
    overbought = params.get("breadth_overbought", 75)

    ax.axhline(y=oversold, color=COLORS["long"], linestyle="--",
               alpha=0.6, label=f"Oversold ({oversold}%)")
    ax.axhline(y=overbought, color=COLORS["short"], linestyle="--",
               alpha=0.6, label=f"Overbought ({overbought}%)")
    ax.axhline(y=50, color=COLORS["neutral"], linestyle=":", alpha=0.4)

    # Fill zones
    ax.fill_between(df.index, 0, oversold, alpha=0.05, color=COLORS["long"])
    ax.fill_between(df.index, overbought, 100, alpha=0.05, color=COLORS["short"])

    # Mark entry points
    if "signal" in df.columns:
        signal_diff = df["signal"].diff()
        long_entries = df.index[signal_diff == 1]
        short_entries = df.index[signal_diff == -1]
        # Reindex to avoid key errors
        long_entry_breadth = df.loc[long_entries, "breadth"] if len(long_entries) > 0 else pd.Series()
        short_entry_breadth = df.loc[short_entries, "breadth"] if len(short_entries) > 0 else pd.Series()

        if not long_entry_breadth.empty:
            ax.scatter(long_entry_breadth.index, long_entry_breadth.values,
                       marker="^", color=COLORS["long"], s=60, zorder=5,
                       label="Long Entry")
        if not short_entry_breadth.empty:
            ax.scatter(short_entry_breadth.index, short_entry_breadth.values,
                       marker="v", color=COLORS["short"], s=60, zorder=5,
                       label="Short Entry")

    ax.set_title("Market Breadth Indicator", fontsize=14, fontweight="bold")
    ax.set_ylabel("% of Stocks Above 200 DMA")
    ax.set_ylim(-5, 105)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)


def _plot_monthly_heatmap(ax, result: BacktestResult):
    """Plot monthly returns heatmap."""
    df = result.signals_df
    if df.empty or "net_return" not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Calculate monthly returns
    monthly = df["net_return"].resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    monthly_pivot = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values,
    })

    pivot = monthly_pivot.pivot_table(index="Year", columns="Month",
                                       values="Return", aggfunc="sum")

    if pivot.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return

    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", center=0, annot=True,
                fmt=".1f", linewidths=0.5, cbar_kws={"label": "Return %"})
    ax.set_title("Monthly Returns (%)", fontsize=14, fontweight="bold")


def _plot_trade_distribution(ax, result: BacktestResult):
    """Plot trade PnL distribution histogram."""
    if not result.trades:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center",
                transform=ax.transAxes)
        return

    pnls = [t.pnl_pct for t in result.trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    if winners:
        ax.hist(winners, bins=20, alpha=0.7, color=COLORS["long"],
                label=f"Winners ({len(winners)})")
    if losers:
        ax.hist(losers, bins=20, alpha=0.7, color=COLORS["short"],
                label=f"Losers ({len(losers)})")

    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_title("Trade PnL Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("PnL (%)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def _plot_summary_table(ax, result: BacktestResult):
    """Plot summary statistics as a table."""
    ax.axis("off")

    engine = BacktestEngine()
    summary = engine.summary_dict(result)

    # Format as two-column table
    keys = list(summary.keys())
    mid = len(keys) // 2 + len(keys) % 2

    col1_data = [[k, str(v)] for k, v in list(summary.items())[:mid]]
    col2_data = [[k, str(v)] for k, v in list(summary.items())[mid:]]

    # Pad shorter column
    while len(col2_data) < len(col1_data):
        col2_data.append(["", ""])

    table_data = [[c1[0], c1[1], c2[0], c2[1]]
                  for c1, c2 in zip(col1_data, col2_data)]

    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Value", "Metric", "Value"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(4):
        table[0, j].set_facecolor("#2196F3")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Performance Summary", fontsize=14, fontweight="bold",
                 pad=20)


def plot_comparison_chart(results: list[BacktestResult],
                          save: bool = True) -> str | None:
    """
    Plot side-by-side comparison of multiple indices.
    """
    if not results:
        return None

    n = len(results)
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    fig.suptitle("Cross-Index Strategy Comparison",
                 fontsize=18, fontweight="bold")

    # Panel 1: Normalized equity curves
    ax1 = axes[0]
    for r in results:
        if not r.equity_curve.empty:
            normalized = r.equity_curve / r.equity_curve.iloc[0] * 100
            ax1.plot(normalized.index, normalized.values,
                     linewidth=1.5, label=r.index_name)

    ax1.set_title("Normalized Equity Curves (base = 100)", fontsize=14)
    ax1.set_ylabel("Value")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Performance bar chart
    ax2 = axes[1]
    metrics = ["Annual Return (%)", "Sharpe Ratio", "Max Drawdown (%)",
               "Win Rate (%)"]
    engine = BacktestEngine()
    summaries = [engine.summary_dict(r) for r in results]
    names = [r.index_name for r in results]

    x = np.arange(len(metrics))
    width = 0.8 / n

    for i, (summary, name) in enumerate(zip(summaries, names)):
        values = [summary.get(m, 0) for m in metrics]
        ax2.bar(x + i * width, values, width, label=name, alpha=0.8)

    ax2.set_xticks(x + width * n / 2)
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.set_title("Key Metrics Comparison", fontsize=14)
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filepath = OUTPUT_DIR / "comparison_report.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        return str(filepath)

    plt.close(fig)
    return None


def generate_text_report(result: BacktestResult) -> str:
    """Generate a text-based backtest report."""
    engine = BacktestEngine()
    summary = engine.summary_dict(result)

    lines = [
        "=" * 70,
        f"  BACKTEST REPORT: {result.index_name}",
        "=" * 70,
        "",
    ]

    for key, value in summary.items():
        lines.append(f"  {key:<30s} {str(value):>15s}")

    lines.extend([
        "",
        "-" * 70,
        "  STRATEGY PARAMETERS",
        "-" * 70,
    ])

    for key, value in result.params.items():
        lines.append(f"  {key:<30s} {str(value):>15s}")

    if result.trades:
        lines.extend([
            "",
            "-" * 70,
            "  RECENT TRADES (last 10)",
            "-" * 70,
            f"  {'Entry Date':<12s} {'Exit Date':<12s} {'Dir':>5s} "
            f"{'Entry':>10s} {'Exit':>10s} {'PnL%':>8s} {'Days':>6s}",
        ])

        for trade in result.trades[-10:]:
            direction = "LONG" if trade.direction == 1 else "SHORT"
            lines.append(
                f"  {trade.entry_date.strftime('%Y-%m-%d'):<12s} "
                f"{trade.exit_date.strftime('%Y-%m-%d'):<12s} "
                f"{direction:>5s} "
                f"{trade.entry_price:>10.2f} "
                f"{trade.exit_price:>10.2f} "
                f"{trade.pnl_pct:>+8.2f} "
                f"{trade.holding_days:>6d}"
            )

    lines.append("=" * 70)
    return "\n".join(lines)
