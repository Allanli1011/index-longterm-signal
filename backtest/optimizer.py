"""
Parameter Optimization Module.

Provides grid search and walk-forward optimization to find
the best strategy parameters for each index.

Includes:
1. Grid Search: Exhaustive search over parameter combinations
2. Walk-Forward: Rolling window optimization for robustness
3. Cross-validation: Prevent overfitting with train/test splits
"""
from __future__ import annotations

import logging
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from strategy.breadth import BreadthAnalyzer
from strategy.signals import StrategyParams, generate_signals
from backtest.engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    best_params: StrategyParams
    best_score: float
    best_result: BacktestResult
    all_results: pd.DataFrame  # Grid of all parameter combinations and scores
    walk_forward_results: list[BacktestResult] | None = None


# Parameter search space
DEFAULT_PARAM_GRID = {
    "breadth_oversold": [15, 20, 25, 30],
    "breadth_overbought": [70, 75, 80, 85],
    "exit_neutral_low": [35, 40, 45, 50],
    "exit_neutral_high": [50, 55, 60, 65],
    "momentum_confirm_threshold": [0, 1.0, 2.0, 3.0],
    "mode": ["simple", "momentum_confirmed", "divergence"],
}

# Reduced grid for faster optimization
FAST_PARAM_GRID = {
    "breadth_oversold": [20, 25, 30],
    "breadth_overbought": [70, 75, 80],
    "exit_neutral_low": [40, 50],
    "exit_neutral_high": [50, 60],
    "momentum_confirm_threshold": [0, 2.0],
    "mode": ["simple", "momentum_confirmed"],
}


def grid_search(close_prices: pd.DataFrame,
                index_price: pd.Series,
                param_grid: dict | None = None,
                base_params: StrategyParams | None = None,
                scoring: str = "sharpe",
                n_jobs: int = -1) -> OptimizationResult:
    """
    Exhaustive grid search over parameter combinations.

    Args:
        close_prices: Constituent stock close prices
        index_price: Index price series
        param_grid: Dict of parameter names -> list of values to try
        base_params: Base parameters (non-grid params use these values)
        scoring: Optimization metric ('sharpe', 'calmar', 'return', 'sortino')
        n_jobs: Number of parallel jobs (-1 = all cores)

    Returns:
        OptimizationResult with best parameters and full grid results
    """
    if param_grid is None:
        param_grid = FAST_PARAM_GRID
    if base_params is None:
        base_params = StrategyParams()

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    logger.info(f"Grid search: {len(combinations)} parameter combinations")

    # Build breadth analyzer once (expensive computation)
    analyzer = BreadthAnalyzer(
        close_prices, index_price,
        ma_period=base_params.ma_period,
        momentum_window=base_params.breadth_momentum_window
        if hasattr(base_params, "breadth_momentum_window")
        else 10,
    )
    signals_df = analyzer.get_signals_df()

    # Evaluate each combination
    def _evaluate(combo):
        params = StrategyParams(**{**base_params.to_dict()})
        for name, value in zip(param_names, combo):
            setattr(params, name, value)

        try:
            result_df = generate_signals(signals_df, params)
            engine = BacktestEngine()
            result = engine.run(result_df, params=params.to_dict())
            score = _get_score(result, scoring)
            return {
                **{name: value for name, value in zip(param_names, combo)},
                "score": score,
                "sharpe": result.sharpe_ratio,
                "annual_return": result.annual_return_pct,
                "max_drawdown": result.max_drawdown_pct,
                "win_rate": result.win_rate,
                "total_trades": result.total_trades,
                "profit_factor": result.profit_factor,
            }
        except Exception as e:
            logger.debug(f"Failed combination {combo}: {e}")
            return {
                **{name: value for name, value in zip(param_names, combo)},
                "score": -999,
            }

    # Run in parallel
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_evaluate)(combo) for combo in combinations
    )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("score", ascending=False)

    # Get best parameters
    best_row = results_df.iloc[0]
    best_params = StrategyParams(**{**base_params.to_dict()})
    for name in param_names:
        setattr(best_params, name, best_row[name])

    # Re-run best for full result
    best_signals = generate_signals(signals_df, best_params)
    engine = BacktestEngine()
    best_result = engine.run(best_signals, params=best_params.to_dict())

    return OptimizationResult(
        best_params=best_params,
        best_score=best_row["score"],
        best_result=best_result,
        all_results=results_df,
    )


def walk_forward_optimization(close_prices: pd.DataFrame,
                               index_price: pd.Series,
                               param_grid: dict | None = None,
                               base_params: StrategyParams | None = None,
                               train_years: int = 5,
                               test_years: int = 1,
                               scoring: str = "sharpe",
                               n_jobs: int = -1) -> OptimizationResult:
    """
    Walk-forward optimization to test parameter stability.

    Splits data into rolling train/test windows:
    1. Optimize on train window
    2. Test on out-of-sample test window
    3. Roll forward and repeat

    This gives a more realistic assessment of strategy performance
    and helps detect overfitting.
    """
    if param_grid is None:
        param_grid = FAST_PARAM_GRID
    if base_params is None:
        base_params = StrategyParams()

    # Determine date ranges
    common_dates = close_prices.index.intersection(index_price.index)
    start = common_dates[0]
    end = common_dates[-1]

    total_years = (end - start).days / 365.25
    if total_years < train_years + test_years:
        logger.warning("Insufficient data for walk-forward. Running simple grid search.")
        return grid_search(close_prices, index_price, param_grid,
                           base_params, scoring, n_jobs)

    # Generate windows
    windows = []
    current_start = start
    while True:
        train_end = current_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)

        if test_end > end:
            break

        windows.append({
            "train_start": current_start,
            "train_end": train_end,
            "test_start": train_end,
            "test_end": test_end,
        })

        current_start = current_start + pd.DateOffset(years=test_years)

    logger.info(f"Walk-forward: {len(windows)} windows")

    # Run optimization on each window
    wf_results = []
    best_params_per_window = []

    for i, window in enumerate(windows):
        logger.info(f"Window {i+1}/{len(windows)}: "
                     f"Train {window['train_start'].strftime('%Y-%m-%d')} to "
                     f"{window['train_end'].strftime('%Y-%m-%d')}, "
                     f"Test {window['test_start'].strftime('%Y-%m-%d')} to "
                     f"{window['test_end'].strftime('%Y-%m-%d')}")

        # Slice data for train window
        train_mask = (close_prices.index >= window["train_start"]) & \
                     (close_prices.index < window["train_end"])
        train_prices = close_prices.loc[train_mask]
        train_index = index_price.loc[
            (index_price.index >= window["train_start"]) &
            (index_price.index < window["train_end"])
        ]

        # Optimize on train
        opt_result = grid_search(train_prices, train_index, param_grid,
                                  base_params, scoring, n_jobs)
        best_params_per_window.append(opt_result.best_params)

        # Test on out-of-sample
        test_mask = (close_prices.index >= window["test_start"]) & \
                    (close_prices.index < window["test_end"])
        test_prices = close_prices.loc[test_mask]
        test_index = index_price.loc[
            (index_price.index >= window["test_start"]) &
            (index_price.index < window["test_end"])
        ]

        analyzer = BreadthAnalyzer(test_prices, test_index,
                                    ma_period=opt_result.best_params.ma_period)
        signals_df = analyzer.get_signals_df()
        test_signals = generate_signals(signals_df, opt_result.best_params)
        engine = BacktestEngine()
        test_result = engine.run(test_signals,
                                  index_name=f"Window_{i+1}",
                                  params=opt_result.best_params.to_dict())
        wf_results.append(test_result)

    # Find most frequently optimal parameters
    param_counts = {}
    for params in best_params_per_window:
        key = str(params.to_dict())
        param_counts[key] = param_counts.get(key, 0) + 1

    # Use the overall best (most stable) parameters
    most_common_key = max(param_counts, key=param_counts.get)
    # Find the params object matching the most common
    for params in best_params_per_window:
        if str(params.to_dict()) == most_common_key:
            final_best_params = params
            break
    else:
        final_best_params = best_params_per_window[0]

    # Run full backtest with best params
    analyzer = BreadthAnalyzer(close_prices, index_price,
                                ma_period=final_best_params.ma_period)
    signals_df = analyzer.get_signals_df()
    final_signals = generate_signals(signals_df, final_best_params)
    engine = BacktestEngine()
    final_result = engine.run(final_signals, params=final_best_params.to_dict())

    # Compile walk-forward grid results
    wf_grid = pd.DataFrame([{
        "window": i + 1,
        "train_period": f"{w['train_start'].strftime('%Y-%m-%d')} to {w['train_end'].strftime('%Y-%m-%d')}",
        "test_period": f"{w['test_start'].strftime('%Y-%m-%d')} to {w['test_end'].strftime('%Y-%m-%d')}",
        "oos_sharpe": r.sharpe_ratio,
        "oos_return": r.total_return_pct,
        "oos_drawdown": r.max_drawdown_pct,
        "oos_trades": r.total_trades,
        "best_oversold": p.breadth_oversold,
        "best_overbought": p.breadth_overbought,
        "best_mode": p.mode,
    } for i, (w, r, p) in enumerate(
        zip(windows, wf_results, best_params_per_window)
    )])

    return OptimizationResult(
        best_params=final_best_params,
        best_score=_get_score(final_result, scoring),
        best_result=final_result,
        all_results=wf_grid,
        walk_forward_results=wf_results,
    )


def _get_score(result: BacktestResult, scoring: str) -> float:
    """Extract optimization score from backtest result."""
    scoring_map = {
        "sharpe": result.sharpe_ratio,
        "sortino": result.sortino_ratio,
        "calmar": result.calmar_ratio,
        "return": result.annual_return_pct,
        "risk_adjusted": (
            result.sharpe_ratio * 0.4 +
            result.sortino_ratio * 0.3 +
            result.calmar_ratio * 0.3
        ),
    }
    return scoring_map.get(scoring, result.sharpe_ratio)
