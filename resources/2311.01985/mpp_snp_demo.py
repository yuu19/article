"""Download S&P 500 component data, generate naive forecasts, and run the MPP optimizer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from mpp_pipeline import (
    MPPInputs,
    NormalizedLinearizationMPP,
    compute_forecast_errors,
)

# デフォルトでは時価総額上位のS&P500銘柄を使用する
TICKERS: List[str] = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "TSLA",
    "BRK-B",
    "UNH",
    "JPM",
    "HD",
    "KO",
    "PEP",
    "PG",
    "V",
]


@dataclass
class MPPDemoResult:
    weights: pd.Series
    r2: float
    iterations: int
    scaling_history: List[float]
    returns: pd.DataFrame
    forecasts: pd.DataFrame


def load_monthly_returns(
    tickers: Iterable[str],
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download monthly close prices and convert to percentage returns."""
    data = yf.download(
        list(tickers),
        start=start,
        end=end,
        interval="1mo",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )
    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs("Close", axis=1, level=1)
    else:
        close = data
    returns = close.pct_change().dropna(how="all")
    return returns.dropna(axis=1, how="any")


def naive_forecasts(returns: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """Create simple moving-average forecasts used as a demonstration."""
    rolling_mean = returns.shift(1).rolling(lookback).mean()
    return rolling_mean.dropna(how="any")


def run_demo(
    tickers: Optional[Iterable[str]] = None,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    lookback: int = 3,
    weight_cap: float = 0.2,
    rho: float = -0.05,
    verbose: bool = False,
) -> MPPDemoResult:
    """Run the MPP optimizer using naive forecasts for the selected tickers."""
    tickers = list(tickers) if tickers is not None else TICKERS
    monthly_returns = load_monthly_returns(tickers, start=start, end=end)
    forecasts = naive_forecasts(monthly_returns, lookback=lookback)
    aligned_returns = monthly_returns.loc[forecasts.index].dropna(how="any")
    forecasts = forecasts.loc[aligned_returns.index]
    realized = aligned_returns.to_numpy()
    predicted = forecasts.to_numpy()
    errors = compute_forecast_errors(predicted, realized)
    expected = predicted.mean(axis=0)
    inputs = MPPInputs(
        returns=realized,
        forecast_errors=errors,
        expected_returns=expected,
        weight_cap=weight_cap,
        rho=rho,
    )
    optimizer = NormalizedLinearizationMPP(inputs)
    weights = optimizer.solve(verbose=verbose)
    r2 = 1.0 - float(weights @ inputs.ete @ weights) / float(weights @ inputs.rtr @ weights)
    weight_series = pd.Series(weights, index=aligned_returns.columns, name="MPP Weight")
    return MPPDemoResult(
        weights=weight_series.sort_values(ascending=False),
        r2=r2,
        iterations=optimizer.iterations_,
        scaling_history=optimizer.scaling_history_,
        returns=aligned_returns,
        forecasts=forecasts,
    )


if __name__ == "__main__":
    result = run_demo(verbose=True)
    print("MPP R^2:", f"{result.r2:.4f}")
    print("Iterations:", result.iterations)
    print("Scaling history:", [f"{s:.6f}" for s in result.scaling_history])
    print("Top 5 weights:")
    print(result.weights.head())
