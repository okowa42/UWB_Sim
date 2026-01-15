from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AccuracyMetrics:
    rmse_m: float
    mae_m: float
    p95_m: float


def compute_position_errors(true_pos: np.ndarray, est_pos: np.ndarray) -> np.ndarray:
    """
    true_pos: (T,2)
    est_pos:  (T,2)
    returns:  errors (T,)
    """
    e = est_pos - true_pos
    return np.linalg.norm(e, axis=1)


def summarize_errors(errors: np.ndarray) -> AccuracyMetrics:
    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))
    p95 = float(np.percentile(errors, 95))
    return AccuracyMetrics(rmse_m=rmse, mae_m=mae, p95_m=p95)
