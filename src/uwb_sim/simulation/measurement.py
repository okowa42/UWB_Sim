from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RangeMeasurements:
    """
    ranges[t, i] = measured distance from position at time t to anchor i.
    """
    ranges: np.ndarray  # (T, N)
    sigma_m: float


def simulate_ranges_los(
    positions: np.ndarray,   # (T,2)
    anchors: np.ndarray,     # (N,2)
    sigma_m: float,
    seed: Optional[int] = None,
) -> RangeMeasurements:
    rng = np.random.default_rng(seed)
    # True ranges
    diff = positions[:, None, :] - anchors[None, :, :]  # (T,N,2)
    true_ranges = np.linalg.norm(diff, axis=2)          # (T,N)
    noise = rng.normal(loc=0.0, scale=sigma_m, size=true_ranges.shape)
    measured = true_ranges + noise
    return RangeMeasurements(ranges=measured, sigma_m=sigma_m)
