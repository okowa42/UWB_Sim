from __future__ import annotations

import numpy as np


def gdop_2d(xy: np.ndarray, anchors: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute 2D GDOP for range-based multilateration at a single position.

    xy: (2,)
    anchors: (N,2), N>=3
    returns: scalar GDOP (float). Larger = worse geometry.

    Model:
      r_i = ||xy - a_i||
      H_i = [ (x-ax)/r_i, (y-ay)/r_i ]
      GDOP = sqrt(trace((H^T H)^-1))
    """
    x, y = float(xy[0]), float(xy[1])
    A = anchors.astype(float)

    dx = x - A[:, 0]
    dy = y - A[:, 1]
    r = np.sqrt(dx * dx + dy * dy) + eps

    H = np.stack([dx / r, dy / r], axis=1)  # (N,2)
    G = H.T @ H  # (2,2)

    # Invert 2x2 safely
    det = float(G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0])
    if abs(det) < eps:
        return float("inf")

    invG = (1.0 / det) * np.array([[G[1, 1], -G[0, 1]], [-G[1, 0], G[0, 0]]], dtype=float)
    return float(np.sqrt(np.trace(invG)))


def gdop_grid_2d(
    area: dict,
    anchors: np.ndarray,
    grid_n: int = 51,
    margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate GDOP on a regular grid over the area.

    area: {xmin,xmax,ymin,ymax}
    grid_n: number of samples along each axis
    margin: avoid evaluating exactly on borders if desired

    returns:
      xs: (grid_n,)
      ys: (grid_n,)
      Z:  (grid_n, grid_n) where Z[j,i] corresponds to (xs[i], ys[j])
    """
    xmin = float(area["xmin"]) + float(margin)
    xmax = float(area["xmax"]) - float(margin)
    ymin = float(area["ymin"]) + float(margin)
    ymax = float(area["ymax"]) - float(margin)

    xs = np.linspace(xmin, xmax, grid_n)
    ys = np.linspace(ymin, ymax, grid_n)

    Z = np.empty((grid_n, grid_n), dtype=float)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            Z[j, i] = gdop_2d(np.array([x, y]), anchors)

    return xs, ys, Z


def summarize_gdop(Z: np.ndarray) -> dict:
    """
    Summarize GDOP grid values (ignoring inf).
    """
    z = Z[np.isfinite(Z)]
    if z.size == 0:
        return {
            "gdop_mean": float("inf"),
            "gdop_p95": float("inf"),
            "gdop_max": float("inf"),
        }
    return {
        "gdop_mean": float(np.mean(z)),
        "gdop_p95": float(np.quantile(z, 0.95)),
        "gdop_max": float(np.max(z)),
    }
