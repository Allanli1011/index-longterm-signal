"""
Index Futures Breadth Strategy - Main Entry Point.

Usage:
    # Run backtest for a single index
    python main.py --index SP500

    # Run optimization for a single index
    python main.py --index SP500 --optimize

    # Run backtest for all indices
    python main.py --all

    # Run full optimization for all indices
    python main.py --all --optimize

    # Walk-forward optimization
    python main.py --index SP500 --walk-forward

    # Specify date range
    python main.py --index SP500 --start 2010-01-01 --end 2024-12-31

    # Quick mode (fewer parameter combinations)
    python main.py --all --optimize --quick
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

import pandas as pd

from config.indices import INDEX_CONFIG, DEFAULT_STRATEGY_PARAMS
from data.constituents import get_constituents
from data.price_data import get_index_price, get_bulk_close_prices
from strategy.breadth import BreadthAnalyzer
from strategy.signals import StrategyParams, generate_signals
from backtest.engine import BacktestEngine, BacktestResult
from backtest.optimizer import (
    grid_search, walk_forward_optimization,
    FAST_PARAM_GRID, DEFAULT_PARAM_GRID,
)
from analysis.visualization import (
    plot_backtest_report, plot_comparison_chart, generate_text_report,
)

OUTPUT_DIR = Path("output")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_single_backtest(index_key: str, params: StrategyParams | None = None,
                        start: str = "2005-01-01",
                        end: str | None = None) -> BacktestResult | None:
    """
    Run a complete backtest for a single index.

    Steps:
    1. Fetch constituent tickers
    2. Download price data (constituents + index)
    3. Calculate breadth indicators
    4. Generate trading signals
    5. Run backtest
    6. Generate report
    """
    config = INDEX_CONFIG.get(index_key)
    if config is None:
        logger.error(f"Unknown index: {index_key}")
        return None

    name = config["name"]
    logger.info(f"{'='*60}")
    logger.info(f"Running backtest for {name} ({index_key})")
    logger.info(f"{'='*60}")

    # Step 1: Get constituents
    logger.info("Step 1: Fetching constituent tickers...")
    tickers = get_constituents(config["constituent_source"])
    if not tickers:
        logger.error(f"No constituents found for {index_key}")
        return None
    logger.info(f"  Got {len(tickers)} constituent tickers")

    # Step 2: Download price data
    price_source = config.get("price_source", "auto")
    logger.info(f"Step 2: Downloading price data (source={price_source})...")
    close_prices = get_bulk_close_prices(tickers, start=start, end=end,
                                         source=price_source)
    if close_prices.empty:
        logger.error("Failed to download constituent prices")
        return None
    logger.info(f"  Got prices for {len(close_prices.columns)} stocks, "
                f"{len(close_prices)} trading days")

    # Get index price
    index_data = get_index_price(config["index_ticker"], start=start, end=end,
                                  source=price_source)
    if index_data is None or index_data.empty:
        logger.error(f"Failed to download index price for {config['index_ticker']}")
        return None
    index_price = index_data["Close"]
    logger.info(f"  Index price: {len(index_price)} days")

    # Step 3: Calculate breadth
    if params is None:
        params = StrategyParams()
    logger.info(f"Step 3: Calculating breadth indicators (MA={params.ma_period})...")
    analyzer = BreadthAnalyzer(
        close_prices, index_price,
        ma_period=params.ma_period,
        momentum_window=params.breadth_momentum_window,
        divergence_lookback=params.divergence_lookback,
    )
    signals_df = analyzer.get_signals_df()
    logger.info(f"  Breadth calculated: {len(signals_df)} data points")

    if not signals_df.empty:
        latest = signals_df.iloc[-1]
        logger.info(f"  Latest breadth: {latest['breadth']:.1f}%")
        logger.info(f"  Latest momentum: {latest['momentum']:.2f}")

    # Step 4: Generate signals
    logger.info(f"Step 4: Generating signals (mode={params.mode})...")
    result_df = generate_signals(signals_df, params)
    long_days = (result_df["signal"] == 1).sum()
    short_days = (result_df["signal"] == -1).sum()
    flat_days = (result_df["signal"] == 0).sum()
    logger.info(f"  Long: {long_days} days, Short: {short_days} days, "
                f"Flat: {flat_days} days")

    # Step 5: Run backtest
    logger.info("Step 5: Running backtest...")
    engine = BacktestEngine()
    result = engine.run(result_df, index_name=name, params=params.to_dict())

    # Step 6: Report
    report = generate_text_report(result)
    logger.info(f"\n{report}")

    try:
        filepath = plot_backtest_report(result, save=True)
        if filepath:
            logger.info(f"  Chart saved to: {filepath}")
    except Exception as e:
        logger.warning(f"  Could not generate chart: {e}")

    return result


def run_optimization(index_key: str, walk_forward: bool = False,
                     quick: bool = True, start: str = "2005-01-01",
                     end: str | None = None,
                     scoring: str = "sharpe") -> dict | None:
    """
    Run parameter optimization for a single index.
    """
    config = INDEX_CONFIG.get(index_key)
    if config is None:
        logger.error(f"Unknown index: {index_key}")
        return None

    name = config["name"]
    logger.info(f"{'='*60}")
    logger.info(f"Optimizing parameters for {name} ({index_key})")
    logger.info(f"{'='*60}")

    # Get data
    price_source = config.get("price_source", "auto")
    tickers = get_constituents(config["constituent_source"])
    if not tickers:
        logger.error(f"No constituents found for {index_key}")
        return None

    close_prices = get_bulk_close_prices(tickers, start=start, end=end,
                                          source=price_source)
    if close_prices.empty:
        logger.error("Failed to download prices")
        return None

    index_data = get_index_price(config["index_ticker"], start=start, end=end,
                                  source=price_source)
    if index_data is None or index_data.empty:
        logger.error("Failed to download index price")
        return None
    index_price = index_data["Close"]

    param_grid = FAST_PARAM_GRID if quick else DEFAULT_PARAM_GRID

    if walk_forward:
        logger.info("Running walk-forward optimization...")
        opt_result = walk_forward_optimization(
            close_prices, index_price,
            param_grid=param_grid,
            scoring=scoring,
        )
    else:
        logger.info(f"Running grid search ({len(param_grid)} dimensions)...")
        opt_result = grid_search(
            close_prices, index_price,
            param_grid=param_grid,
            scoring=scoring,
        )

    # Report results
    best = opt_result.best_params
    logger.info(f"\nBest Parameters for {name}:")
    logger.info(f"  Mode:                  {best.mode}")
    logger.info(f"  Oversold Threshold:    {best.breadth_oversold}%")
    logger.info(f"  Overbought Threshold:  {best.breadth_overbought}%")
    logger.info(f"  Exit Neutral Low:      {best.exit_neutral_low}%")
    logger.info(f"  Exit Neutral High:     {best.exit_neutral_high}%")
    logger.info(f"  Momentum Threshold:    {best.momentum_confirm_threshold}")
    logger.info(f"  Best Score ({scoring}):  {opt_result.best_score:.4f}")

    # Show top 5 combinations
    logger.info("\nTop 5 Parameter Combinations:")
    top5 = opt_result.all_results.head(5)
    logger.info(f"\n{top5.to_string()}")

    # Generate full backtest report with best params
    report = generate_text_report(opt_result.best_result)
    logger.info(f"\n{report}")

    try:
        filepath = plot_backtest_report(opt_result.best_result, save=True)
        if filepath:
            logger.info(f"Chart saved to: {filepath}")
    except Exception as e:
        logger.warning(f"Could not generate chart: {e}")

    return {
        "index": index_key,
        "name": name,
        "best_params": best.to_dict(),
        "best_score": opt_result.best_score,
        "sharpe": opt_result.best_result.sharpe_ratio,
        "annual_return": opt_result.best_result.annual_return_pct,
        "max_drawdown": opt_result.best_result.max_drawdown_pct,
    }


def run_all(optimize: bool = False, walk_forward: bool = False,
            quick: bool = True, start: str = "2005-01-01",
            end: str | None = None, indices: list[str] | None = None):
    """
    Run backtest or optimization for all (or selected) indices.
    """
    index_keys = indices or list(INDEX_CONFIG.keys())
    results = []
    opt_results = []

    for key in index_keys:
        try:
            if optimize:
                opt = run_optimization(key, walk_forward=walk_forward,
                                        quick=quick, start=start, end=end)
                if opt:
                    opt_results.append(opt)
                    # Also run backtest with optimized params
                    best_params = StrategyParams.from_dict(opt["best_params"])
                    result = run_single_backtest(key, params=best_params,
                                                  start=start, end=end)
                    if result:
                        results.append(result)
            else:
                result = run_single_backtest(key, start=start, end=end)
                if result:
                    results.append(result)
        except Exception as e:
            logger.error(f"Failed for {key}: {e}", exc_info=True)
            continue

    # Cross-index comparison
    if len(results) > 1:
        try:
            filepath = plot_comparison_chart(results, save=True)
            if filepath:
                logger.info(f"\nComparison chart saved to: {filepath}")
        except Exception as e:
            logger.warning(f"Could not generate comparison chart: {e}")

    # Save optimization results
    if opt_results:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        opt_file = OUTPUT_DIR / "optimization_results.json"
        with open(opt_file, "w") as f:
            json.dump(opt_results, f, indent=2, default=str)
        logger.info(f"\nOptimization results saved to: {opt_file}")

        # Print summary table
        logger.info("\n" + "=" * 80)
        logger.info("OPTIMIZATION SUMMARY - BEST PARAMETERS PER INDEX")
        logger.info("=" * 80)
        summary_df = pd.DataFrame(opt_results)
        logger.info(f"\n{summary_df.to_string()}")

    # Print performance comparison
    if results:
        engine = BacktestEngine()
        comparison = pd.DataFrame([engine.summary_dict(r) for r in results])
        logger.info("\n" + "=" * 80)
        logger.info("PERFORMANCE COMPARISON")
        logger.info("=" * 80)
        logger.info(f"\n{comparison.to_string()}")

    return results, opt_results


def main():
    parser = argparse.ArgumentParser(
        description="Index Futures Breadth Strategy Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--index", "-i", type=str,
                        help="Index key to backtest (e.g., SP500, DAX)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Run for all configured indices")
    parser.add_argument("--indices", nargs="+",
                        help="Run for specific indices (e.g., SP500 DAX NIKKEI225)")
    parser.add_argument("--optimize", "-o", action="store_true",
                        help="Run parameter optimization")
    parser.add_argument("--walk-forward", "-wf", action="store_true",
                        help="Use walk-forward optimization")
    parser.add_argument("--quick", "-q", action="store_true", default=True,
                        help="Use reduced parameter grid (default: True)")
    parser.add_argument("--full", "-f", action="store_true",
                        help="Use full parameter grid")
    parser.add_argument("--start", "-s", type=str, default="2005-01-01",
                        help="Backtest start date (default: 2005-01-01)")
    parser.add_argument("--end", "-e", type=str, default=None,
                        help="Backtest end date (default: today)")
    parser.add_argument("--scoring", type=str, default="sharpe",
                        choices=["sharpe", "sortino", "calmar", "return",
                                 "risk_adjusted"],
                        help="Optimization scoring metric")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all available indices")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.list:
        print("\nAvailable Indices:")
        print("-" * 70)
        for key, config in INDEX_CONFIG.items():
            print(f"  {key:<15s} {config['name']:<35s} ({config['country']})")
            print(f"  {'':15s} {config['description']}")
        return

    quick = not args.full

    if args.index:
        if args.optimize:
            run_optimization(args.index, walk_forward=args.walk_forward,
                             quick=quick, start=args.start, end=args.end,
                             scoring=args.scoring)
        else:
            run_single_backtest(args.index, start=args.start, end=args.end)

    elif args.all or args.indices:
        run_all(optimize=args.optimize, walk_forward=args.walk_forward,
                quick=quick, start=args.start, end=args.end,
                indices=args.indices)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python main.py --list")
        print("  python main.py --index SP500")
        print("  python main.py --index SP500 --optimize")
        print("  python main.py --all --optimize --quick")
        print("  python main.py --indices SP500 DAX NIKKEI225 --optimize")


if __name__ == "__main__":
    main()
