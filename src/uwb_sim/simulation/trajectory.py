from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Trajectory2D:
    """Holds 2D trajectory positions over time."""
    positions: np.ndarray  # shape (T,2)
    dt: float


def generate_lawnmower(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    dt: float,
    steps: int,
    speed_mps: float,
    lane_spacing_m: float = 10.0,
    margin_m: float = 5.0,
) -> Trajectory2D:
    """
    Simple lawnmower (back-and-forth) path within the area.
    Deterministic and suitable for debugging.
    """
    # Define inner bounds
    x0, x1 = xmin + margin_m, xmax - margin_m
    y0, y1 = ymin + margin_m, ymax - margin_m

    # Generate waypoints for lanes
    lanes = np.arange(y0, y1 + 1e-9, lane_spacing_m)
    if lanes.size < 2:
        lanes = np.array([y0, y1])

    waypoints = []
    direction = 1
    for y in lanes:
        if direction > 0:
            waypoints.append((x0, y))
            waypoints.append((x1, y))
        else:
            waypoints.append((x1, y))
            waypoints.append((x0, y))
        direction *= -1

    waypoints = np.array(waypoints, dtype=float)

    # Convert waypoints to sampled positions at constant speed
    positions = np.zeros((steps, 2), dtype=float)
    positions[0] = waypoints[0]
    wp_idx = 1
    step_len = speed_mps * dt

    for t in range(1, steps):
        current = positions[t - 1]
        target = waypoints[wp_idx]
        vec = target - current
        dist = float(np.linalg.norm(vec))
        if dist < 1e-9:
            wp_idx = min(wp_idx + 1, len(waypoints) - 1)
            positions[t] = current
            continue

        move = min(step_len, dist)
        positions[t] = current + vec * (move / dist)

        # Advance waypoint if reached
        if move >= dist - 1e-9 and wp_idx < len(waypoints) - 1:
            wp_idx += 1

    return Trajectory2D(positions=positions, dt=dt)
