from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class Area2D:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def center(self) -> Tuple[float, float]:
        return (self.xmin + self.width / 2.0, self.ymin + self.height / 2.0)


def _clip_to_area(points: np.ndarray, area: Area2D) -> np.ndarray:
    """Clip points to be inside the area (safety)."""
    pts = points.copy()
    pts[:, 0] = np.clip(pts[:, 0], area.xmin, area.xmax)
    pts[:, 1] = np.clip(pts[:, 1], area.ymin, area.ymax)
    return pts


def generate_anchors(
    area: Area2D,
    pattern: str,
    count: int,
    margin_m: float = 5.0,
    radius_m: float = 40.0,
    grid_nx: int = 2,
    grid_ny: int = 2,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate 2D anchor positions (N, 2).
    Supported patterns:
      - corners: 4 corners with margin (count ignored; output=4)
      - perimeter: equally spaced on rectangle perimeter (count used; >=3 recommended)
      - circle: equally spaced on a circle around center (count used; >=3 recommended)
      - grid: grid points inside area (uses grid_nx * grid_ny; count ignored)
      - random: uniform random inside area with margin (count used)

    Returns:
      anchors: np.ndarray shape (N,2)
    """
    pattern = pattern.lower().strip()

    if pattern == "corners":
        x0, x1 = area.xmin + margin_m, area.xmax - margin_m
        y0, y1 = area.ymin + margin_m, area.ymax - margin_m
        anchors = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
        return anchors

    if pattern == "perimeter":
        if count < 3:
            raise ValueError("perimeter pattern requires count >= 3")
        # Perimeter parameterization: walk around rectangle edges
        x0, x1 = area.xmin + margin_m, area.xmax - margin_m
        y0, y1 = area.ymin + margin_m, area.ymax - margin_m
        # Total perimeter length
        L = 2 * ((x1 - x0) + (y1 - y0))
        ts = np.linspace(0.0, L, num=count, endpoint=False)
        pts = []
        for t in ts:
            if t < (x1 - x0):  # bottom edge
                pts.append((x0 + t, y0))
            elif t < (x1 - x0) + (y1 - y0):  # right edge
                tt = t - (x1 - x0)
                pts.append((x1, y0 + tt))
            elif t < 2 * (x1 - x0) + (y1 - y0):  # top edge
                tt = t - ((x1 - x0) + (y1 - y0))
                pts.append((x1 - tt, y1))
            else:  # left edge
                tt = t - (2 * (x1 - x0) + (y1 - y0))
                pts.append((x0, y1 - tt))
        anchors = np.array(pts, dtype=float)
        return anchors

    if pattern == "circle":
        if count < 3:
            raise ValueError("circle pattern requires count >= 3")
        cx, cy = area.center
        angles = np.linspace(0.0, 2 * np.pi, num=count, endpoint=False)
        xs = cx + radius_m * np.cos(angles)
        ys = cy + radius_m * np.sin(angles)
        anchors = np.stack([xs, ys], axis=1).astype(float)
        anchors = _clip_to_area(anchors, area)
        return anchors

    if pattern == "grid":
        if grid_nx < 2 or grid_ny < 2:
            raise ValueError("grid pattern requires grid_nx>=2 and grid_ny>=2")
        xs = np.linspace(area.xmin + margin_m, area.xmax - margin_m, grid_nx)
        ys = np.linspace(area.ymin + margin_m, area.ymax - margin_m, grid_ny)
        X, Y = np.meshgrid(xs, ys)
        anchors = np.stack([X.ravel(), Y.ravel()], axis=1).astype(float)
        return anchors

    if pattern == "random":
        if count < 1:
            raise ValueError("random pattern requires count >= 1")
        rng = np.random.default_rng(seed)
        xs = rng.uniform(area.xmin + margin_m, area.xmax - margin_m, size=count)
        ys = rng.uniform(area.ymin + margin_m, area.ymax - margin_m, size=count)
        anchors = np.stack([xs, ys], axis=1).astype(float)
        return anchors

    raise ValueError(f"Unknown anchor pattern: {pattern}")
