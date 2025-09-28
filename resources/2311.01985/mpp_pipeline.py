"""Utilities inspired by Pinelis & Ruppert (2023) for maximizing portfolio predictability."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - cvxpy may be missing by default
    cp = None


@dataclass
class MPPInputs:
    """Container for matrices needed by the MPP optimizer."""

    returns: np.ndarray  # shape (T, n)
    forecast_errors: np.ndarray  # shape (T, n)
    expected_returns: np.ndarray  # shape (n,)
    weight_cap: Optional[float] = 0.3
    rho: float = -0.1
    lower_bound: float = 0.0

    def __post_init__(self) -> None:
        if self.returns.shape != self.forecast_errors.shape:
            raise ValueError("returns and forecast_errors must share shape")
        if self.returns.ndim != 2:
            raise ValueError("returns must be 2D (time x assets)")
        if self.expected_returns.shape[0] != self.returns.shape[1]:
            raise ValueError("expected_returns length must match asset count")

    @property
    def ete(self) -> np.ndarray:
        return self.forecast_errors.T @ self.forecast_errors

    @property
    def rtr(self) -> np.ndarray:
        return self.returns.T @ self.returns


class NormalizedLinearizationMPP:
    """Implements the Normalized Linearization Algorithm (Gotoh & Fujisawa, 2012)."""

    def __init__(
        self,
        data: MPPInputs,
        extra_linear_constraints: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
        solver: Optional[str] = None,
    ) -> None:
        self.data = data
        self.extra_linear_constraints: List[Tuple[np.ndarray, np.ndarray]] = (
            list(extra_linear_constraints) if extra_linear_constraints else []
        )
        self.solver = solver
        self.iterations_: int = 0
        self.scaling_history_: List[float] = []
        if cp is None:
            raise ImportError(
                "cvxpy is required for the normalized linearization routine."
            )

    def solve(
        self,
        tol: float = 1e-3,
        max_iter: int = 50,
        verbose: bool = False,
    ) -> np.ndarray:
        """Run the NLA iterations and return unit-sum portfolio weights."""
        returns = self.data.returns
        ete = self.data.ete
        mu = self.data.expected_returns
        n_assets = returns.shape[1]
        y_prev = np.full(n_assets, 1.0 / math.sqrt(n_assets))
        u_prev = returns @ y_prev
        norm_u = np.linalg.norm(u_prev)
        if norm_u == 0:
            raise ValueError("Initial projection is zero; provide richer data")
        u_prev /= norm_u
        self.scaling_history_.clear()
        for iteration in range(max_iter):
            y_var = cp.Variable(n_assets)
            objective = cp.Minimize(cp.quad_form(y_var, ete))
            constraints: List = []
            if self.data.weight_cap is not None:
                constraints.append(y_var <= self.data.weight_cap)
            constraints.append(y_var >= self.data.lower_bound)
            constraints.append(cp.sum(y_var) >= 0)
            constraints.append((returns.T @ u_prev) @ y_var == 1)
            if self.data.rho is not None:
                constraints.append(mu @ y_var >= self.data.rho * cp.sum(y_var))
            for A, b in self.extra_linear_constraints:
                constraints.append(A @ y_var <= b)
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=self.solver)
            if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise RuntimeError(f"NLA QP failed at iter {iteration}: {problem.status}")
            y_hat = y_var.value
            u_hat = returns @ y_hat
            scaling = math.sqrt(float(u_hat.T @ u_hat))
            if scaling == 0:
                raise RuntimeError("Degenerate iterate encountered (zero norm)")
            self.scaling_history_.append(scaling)
            if verbose:
                print(f"iteration={iteration} scaling={scaling:.6f}")
            y_curr = y_hat / scaling
            u_curr = u_hat / scaling
            if scaling <= 1 + tol:
                weights = y_curr / np.sum(y_curr)
                if np.any(weights < -1e-6):
                    raise RuntimeError("Non-feasible weights detected. Adjust bounds.")
                self.iterations_ = iteration + 1
                return weights
            y_prev, u_prev = y_curr, u_curr
        self.iterations_ = max_iter
        raise RuntimeError("NLA did not converge within max_iter")


def compute_forecast_errors(predicted: np.ndarray, realized: np.ndarray) -> np.ndarray:
    """Return forecast errors aligned with the paper's definition."""
    if predicted.shape != realized.shape:
        raise ValueError("predicted and realized arrays must match")
    return realized - predicted


def reward_risk_timing(
    mpp_returns: Sequence[float],
    mpp_volatility: Sequence[float],
    risk_free: Sequence[float],
    risk_aversion: float = 4.0,
) -> np.ndarray:
    """Compute Kelly-style timing weights for the MPP."""
    mpp_returns = np.asarray(mpp_returns)
    mpp_volatility = np.asarray(mpp_volatility)
    risk_free = np.asarray(risk_free)
    if not (mpp_returns.shape == mpp_volatility.shape == risk_free.shape):
        raise ValueError("Inputs must share shape")
    excess = mpp_returns - risk_free
    denom = risk_aversion * (mpp_volatility**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(denom > 0, excess / denom, 0.0)
    return np.clip(weights, -5.0, 5.0)


def decile_assignment(scores: Sequence[float], n_deciles: int = 10) -> np.ndarray:
    """Assign assets to deciles based on score ranks (0-indexed)."""
    scores = np.asarray(scores)
    ranks = scores.argsort().argsort()
    quantile = np.ceil((ranks + 1) * n_deciles / scores.size) - 1
    return quantile.astype(int)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    periods, assets = 120, 50
    returns = rng.normal(scale=0.02, size=(periods, assets))
    predicted = returns + rng.normal(scale=0.01, size=(periods, assets))
    errors = compute_forecast_errors(predicted, returns)
    expected = predicted.mean(axis=0)
    inputs = MPPInputs(returns=returns, forecast_errors=errors, expected_returns=expected)
    try:
        optimizer = NormalizedLinearizationMPP(inputs)
        weights = optimizer.solve(verbose=True)
        print("MPP weights", weights)
        print("Iterations", optimizer.iterations_)
        print("Scalings", optimizer.scaling_history_)
    except ImportError:
        print("Install cvxpy to run the optimizer")
