from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class WLSResult:
    position: np.ndarray          # (2,)
    converged: bool
    iterations: int
    final_cost: float


def wls_gauss_newton_2d(
    anchors: np.ndarray,          # (N,2)
    ranges: np.ndarray,           # (N,)
    sigma_m: float,
    x0: Optional[np.ndarray] = None,
    max_iterations: int = 20,
    tol: float = 1e-6,
) -> WLSResult:
    """
    Weighted nonlinear least squares for 2D multilateration using Gauss-Newton.

    Minimize: sum_i ((||x - a_i|| - r_i)^2 / sigma^2)
    """
    N = anchors.shape[0]
    if N < 3:
        raise ValueError("2D multilateration requires at least 3 anchors.")

    w = 1.0 / (sigma_m ** 2)

    if x0 is None:
        x = anchors.mean(axis=0).astype(float)
    else:
        x = np.array(x0, dtype=float)

    converged = False
    last_cost = None

    for it in range(1, max_iterations + 1):
        diff = x[None, :] - anchors           # (N,2)
        d = np.linalg.norm(diff, axis=1)      # (N,)
        d = np.maximum(d, 1e-9)

        residual = d - ranges                 # (N,)

        # Jacobian: dr_i/dx = (x - a_i)/||x-a_i||
        J = diff / d[:, None]                 # (N,2)

        # Gauss-Newton step: (J^T W J) dx = - J^T W r
        # Here W = w * I (scalar) because identical sigma
        A = (J.T @ J) * w
        b = -(J.T @ residual) * w

        try:
            dx = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return WLSResult(position=x, converged=False, iterations=it, final_cost=float("inf"))

        x = x + dx

        cost = float((residual @ residual) * w)
        if last_cost is not None and abs(last_cost - cost) < tol:
            converged = True
            last_cost = cost
            return WLSResult(position=x, converged=True, iterations=it, final_cost=cost)

        if float(np.linalg.norm(dx)) < tol:
            converged = True
            last_cost = cost
            return WLSResult(position=x, converged=True, iterations=it, final_cost=cost)

        last_cost = cost

    return WLSResult(position=x, converged=converged, iterations=max_iterations, final_cost=float(last_cost or 0.0))
