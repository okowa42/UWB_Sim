from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from uwb_sim.estimation.multilateration import wls_gauss_newton_2d
from uwb_sim.metrics.accuracy import compute_position_errors, summarize_errors, AccuracyMetrics
from uwb_sim.metrics.profiler import Timer, TimerStats
from uwb_sim.simulation.measurement import simulate_ranges_los
from uwb_sim.simulation.trajectory import generate_lawnmower


@dataclass(frozen=True)
class SimulationResult:
    true_positions: np.ndarray   # (T,2)
    est_positions: np.ndarray    # (T,2)
    errors_m: np.ndarray         # (T,)
    metrics: AccuracyMetrics
    timing: dict | None = None   # profiling info (optional)


def run_single_simulation(
    area: dict,
    sim_time: dict,
    traj_cfg: dict,
    anchors: np.ndarray,
    meas_cfg: dict,
    est_cfg: dict,
    seed: Optional[int] = None,
    profile: bool = False,
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
    timer = Timer()
    step_stats = TimerStats()

    for t in range(steps):
        r = ranges[t]
        if profile:
            timer.start()

        res = wls_gauss_newton_2d(
            anchors=anchors,
            ranges=r,
            sigma_m=sigma,
            x0=x_prev,
            max_iterations=int(est_cfg.get("max_iterations", 20)),
            tol=float(est_cfg.get("tol", 1e-6)),
        )

        if profile:
            step_stats.add(timer.stop())

        est_pos[t] = res.position
        x_prev = res.position


    errors = compute_position_errors(true_pos, est_pos)
    metrics = summarize_errors(errors)

    return SimulationResult(true_positions=true_pos, est_positions=est_pos, errors_m=errors, metrics=metrics)

def run_monte_carlo(
    area: dict,
    sim_time: dict,
    traj_cfg: dict,
    anchors: np.ndarray,
    meas_cfg: dict,
    est_cfg: dict,
    base_seed: int,
    trials: int,
) -> tuple[dict, list[dict]]:
    """
    Run K Monte Carlo trials by varying seed = base_seed + k.

    Returns:
      summary_dict: aggregated statistics
      per_trial_rows: list of each-trial metrics rows
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")

    rmse_list = []
    mae_list = []
    p95_list = []

    per_trial_rows: list[dict] = []

    for k in range(trials):
        seed = base_seed + k
        res = run_single_simulation(
            area=area,
            sim_time=sim_time,
            traj_cfg=traj_cfg,
            anchors=anchors,
            meas_cfg=meas_cfg,
            est_cfg=est_cfg,
            seed=seed,
        )

        row = {
            "trial": k,
            "seed": seed,
            "rmse_m": res.metrics.rmse_m,
            "mae_m": res.metrics.mae_m,
            "p95_m": res.metrics.p95_m,
            "anchors": int(anchors.shape[0]),
            "dt": float(sim_time["dt"]),
            "steps": int(sim_time["steps"]),
            "range_noise_sigma_m": float(meas_cfg["range_noise_sigma_m"]),
        }
        per_trial_rows.append(row)

        rmse_list.append(res.metrics.rmse_m)
        mae_list.append(res.metrics.mae_m)
        p95_list.append(res.metrics.p95_m)

    rmse = np.array(rmse_list, dtype=float)
    mae = np.array(mae_list, dtype=float)
    p95 = np.array(p95_list, dtype=float)

    summary_dict = {
        "trials": trials,
        "base_seed": int(base_seed),
        "anchors": int(anchors.shape[0]),
        "dt": float(sim_time["dt"]),
        "steps": int(sim_time["steps"]),
        "range_noise_sigma_m": float(meas_cfg["range_noise_sigma_m"]),
        "rmse_mean_m": float(rmse.mean()),
        "rmse_std_m": float(rmse.std(ddof=1)) if trials >= 2 else 0.0,
        "mae_mean_m": float(mae.mean()),
        "mae_std_m": float(mae.std(ddof=1)) if trials >= 2 else 0.0,
        "p95_mean_m": float(p95.mean()),
        "p95_std_m": float(p95.std(ddof=1)) if trials >= 2 else 0.0,
    }

    timing = None
    if profile:
        timing = {
            "wls_step_mean_s": step_stats.mean_s,
            "wls_step_min_s": step_stats.min_s if step_stats.n > 0 else 0.0,
            "wls_step_max_s": step_stats.max_s if step_stats.n > 0 else 0.0,
            "wls_steps_measured": step_stats.n,
            "wls_total_s": step_stats.total_s,
        }

    return SimulationResult(
        true_positions=true_pos,
        est_positions=est_pos,
        errors_m=errors,
        metrics=metrics,
        timing=timing,
    )
