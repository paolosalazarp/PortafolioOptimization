"""
Utility functions for portfolio optimization and efficient frontier analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize


def download_price_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.

    Parameters
    ----------
    tickers : list of str
        Stock ticker symbols.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted closing prices indexed by date.
    """
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    return data.dropna()


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily percentage returns from price data.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices.

    Returns
    -------
    pd.DataFrame
        DataFrame of daily returns.
    """
    return prices.pct_change().dropna()


def portfolio_performance(weights: np.ndarray, mean_returns: pd.Series,
                          cov_matrix: pd.DataFrame,
                          trading_days: int = 252) -> tuple:
    """
    Calculate annualised portfolio return, volatility, and Sharpe ratio.

    Parameters
    ----------
    weights : np.ndarray
        Asset weights (must sum to 1).
    mean_returns : pd.Series
        Mean daily returns for each asset.
    cov_matrix : pd.DataFrame
        Covariance matrix of daily returns.
    trading_days : int, optional
        Number of trading days per year (default 252).

    Returns
    -------
    tuple
        (annualised_return, annualised_volatility, sharpe_ratio)
    """
    ret = np.dot(weights, mean_returns) * trading_days
    vol = np.sqrt(weights @ cov_matrix.values @ weights) * np.sqrt(trading_days)
    sharpe = ret / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def simulate_random_portfolios(mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                                num_portfolios: int = 5000,
                                trading_days: int = 252,
                                risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Generate random portfolio weights and compute their performance.

    Parameters
    ----------
    mean_returns : pd.Series
        Mean daily returns for each asset.
    cov_matrix : pd.DataFrame
        Covariance matrix of daily returns.
    num_portfolios : int, optional
        Number of random portfolios to simulate (default 5000).
    trading_days : int, optional
        Number of trading days per year (default 252).
    risk_free_rate : float, optional
        Annual risk-free rate for Sharpe ratio calculation (default 0.0).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Return', 'Volatility', 'Sharpe'] and one row per
        simulated portfolio. Index columns for each asset weight are also included.
    """
    n_assets = len(mean_returns)
    results = np.zeros((num_portfolios, 3 + n_assets))

    for i in range(num_portfolios):
        w = np.random.dirichlet(np.ones(n_assets))
        ret, vol, _ = portfolio_performance(w, mean_returns, cov_matrix, trading_days)
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
        results[i, 0] = ret
        results[i, 1] = vol
        results[i, 2] = sharpe
        results[i, 3:] = w

    columns = ["Return", "Volatility", "Sharpe"] + list(mean_returns.index)
    return pd.DataFrame(results, columns=columns)


def minimize_volatility(mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                        target_return: float,
                        trading_days: int = 252) -> np.ndarray:
    """
    Find portfolio weights that minimise volatility for a given target return.

    Parameters
    ----------
    mean_returns : pd.Series
        Mean daily returns for each asset.
    cov_matrix : pd.DataFrame
        Covariance matrix of daily returns.
    target_return : float
        Desired annualised portfolio return.
    trading_days : int, optional
        Number of trading days per year (default 252).

    Returns
    -------
    np.ndarray
        Optimal weights array.
    """
    n = len(mean_returns)
    args = (mean_returns, cov_matrix, trading_days)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: portfolio_performance(w, *args)[0] - target_return},
    ]
    bounds = tuple((0.0, 1.0) for _ in range(n))
    result = minimize(
        lambda w: portfolio_performance(w, *args)[1],
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    return result.x


def compute_efficient_frontier(mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                                num_points: int = 100,
                                trading_days: int = 252) -> pd.DataFrame:
    """
    Compute the efficient frontier by minimising volatility across a range of returns.

    Parameters
    ----------
    mean_returns : pd.Series
        Mean daily returns for each asset.
    cov_matrix : pd.DataFrame
        Covariance matrix of daily returns.
    num_points : int, optional
        Number of points on the frontier (default 100).
    trading_days : int, optional
        Number of trading days per year (default 252).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'Return' and 'Volatility' columns for each frontier point.
    """
    min_ret = mean_returns.min() * trading_days
    max_ret = mean_returns.max() * trading_days
    target_returns = np.linspace(min_ret, max_ret, num_points)

    frontier = []
    for tr in target_returns:
        try:
            w = minimize_volatility(mean_returns, cov_matrix, tr, trading_days)
            _, vol, _ = portfolio_performance(w, mean_returns, cov_matrix, trading_days)
            frontier.append({"Return": tr, "Volatility": vol})
        except (ValueError, RuntimeError):
            continue

    return pd.DataFrame(frontier)


def plot_efficient_frontier(simulated: pd.DataFrame, frontier: pd.DataFrame,
                             max_sharpe_idx: int = None) -> plt.Figure:
    """
    Plot the efficient frontier alongside randomly simulated portfolios.

    Parameters
    ----------
    simulated : pd.DataFrame
        Output of simulate_random_portfolios().
    frontier : pd.DataFrame
        Output of compute_efficient_frontier().
    max_sharpe_idx : int, optional
        Row index in *simulated* corresponding to the max-Sharpe portfolio.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    scatter = ax.scatter(
        simulated["Volatility"],
        simulated["Return"],
        c=simulated["Sharpe"],
        cmap="viridis",
        alpha=0.5,
        s=10,
        label="Simulated portfolios",
    )
    plt.colorbar(scatter, ax=ax, label="Sharpe Ratio")

    ax.plot(frontier["Volatility"], frontier["Return"],
            "r--", linewidth=2, label="Efficient Frontier")

    if max_sharpe_idx is not None:
        ms = simulated.iloc[max_sharpe_idx]
        ax.scatter(ms["Volatility"], ms["Return"],
                   marker="*", color="gold", s=300, zorder=5, label="Max Sharpe")

    ax.set_xlabel("Annualised Volatility")
    ax.set_ylabel("Annualised Return")
    ax.set_title("Portfolio Optimisation – Efficient Frontier")
    ax.legend()
    plt.tight_layout()
    return fig
