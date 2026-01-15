from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from uwb_sim.estimation.multilateration import wls_gauss_newton_2d
from uwb_sim.metrics.accuracy import compute_position_errors, summarize_errors, AccuracyMetrics
from uwb_sim.simulation.measurement import simulate_ranges_los
from uwb_sim.simulation.trajectory import generate_lawnmower


@dataclass(frozen=True)
class SimulationResult:
    true_positions: np.ndarray   # (T,2)
    est_positions: np.ndarray    # (T,2)
    errors_m: np.ndarray         # (T,)
    metrics: AccuracyMetrics


def run_single_simulation(
    area: dict,
    sim_time: dict,
    traj_cfg: dict,
    anchors: np.ndarray,
    meas_cfg: dict,
    est_cfg: dict,
    seed: Optional[int] = None,
) -> SimulationResult:
    dt = float(sim_time["dt"])
    steps = int(sim_time["steps"])

    # Trajectory
    traj = generate_lawnmower(
        xmin=float(area["xmin"]),
        xmax=float(area["xmax"]),
        ymin=float(area["ymin"]),
        ymax=float(area["ymax"]),
        dt=dt,
        steps=steps,
        speed_mps=float(traj_cfg.get("speed_mps", 1.0)),
    )
    true_pos = traj.positions

    # Ranges
    sigma = float(meas_cfg["range_noise_sigma_m"])
    ranges = simulate_ranges_los(true_pos, anchors, sigma_m=sigma, seed=seed).ranges  # (T,N)

    # Estimation per time step
    est_pos = np.zeros_like(true_pos)
    x_prev = None
    for t in range(steps):
        r = ranges[t]
        res = wls_gauss_newton_2d(
            anchors=anchors,
            ranges=r,
            sigma_m=sigma,
            x0=x_prev,
            max_iterations=int(est_cfg.get("max_iterations", 20)),
            tol=float(est_cfg.get("tol", 1e-6)),
        )
        est_pos[t] = res.position
        x_prev = res.position  # warm-start

    errors = compute_position_errors(true_pos, est_pos)
    metrics = summarize_errors(errors)

    return SimulationResult(true_positions=true_pos, est_positions=est_pos, errors_m=errors, metrics=metrics)
