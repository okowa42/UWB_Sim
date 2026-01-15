from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from uwb_sim.io.config_loader import load_yaml_config
from uwb_sim.io.paths import make_batch_paths
from uwb_sim.scenario.patterns import Area2D, generate_anchors
from uwb_sim.simulation.runner import run_monte_carlo


@dataclass(frozen=True)
class BatchGrid:
    patterns: list[str]
    counts: list[int]
    sigmas: list[float]
    trials: int
    base_seed: int


def _default_grid() -> BatchGrid:
    return BatchGrid(
        patterns=["corners", "perimeter", "circle", "grid", "random"],
        counts=[3, 4, 5, 6, 8, 10],
        sigmas=[0.05, 0.10, 0.20],
        trials=30,
        base_seed=42,
    )


def _make_area(cfg: dict[str, Any]) -> Area2D:
    area_cfg = cfg["simulation"]["area"]
    return Area2D(
        xmin=float(area_cfg["xmin"]),
        xmax=float(area_cfg["xmax"]),
        ymin=float(area_cfg["ymin"]),
        ymax=float(area_cfg["ymax"]),
    )


def _resolve_grid_params(cfg: dict[str, Any], count: int) -> tuple[int, int]:
    """
    For grid pattern, convert "count" (like 4, 9, 16) into (nx, ny).
    If count is not a perfect square, fall back to config values.
    """
    root = int(round(count ** 0.5))
    if root * root == count and root >= 2:
        return root, root
    # fallback
    nx = int(cfg["anchors"]["grid"]["nx"])
    ny = int(cfg["anchors"]["grid"]["ny"])
    return nx, ny


def main():
    cfg = load_yaml_config(Path("configs/default.yaml"))
    area = _make_area(cfg)

    # Base configuration pieces (fixed across batch)
    sim_time = {
        "dt": float(cfg["simulation"]["time"]["dt"]),
        "steps": int(cfg["simulation"]["time"]["steps"]),
    }
    traj_cfg = cfg["trajectory"]
    est_cfg = cfg["estimation"]

    # Batch grid (edit here or later make it a YAML)
    grid = _default_grid()

    # Save executed conditions and summaries
    conditions_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    # Create output paths
    paths = make_batch_paths()

    print(f"[Batch] Output dir: {paths.batch_dir}")

    for pattern in grid.patterns:
        for count in grid.counts:
            # Pattern-specific behavior:
            # - corners ignores count (always 4 anchors). We still log requested count for traceability.
            # - grid uses nx*ny. We map count to square grid when possible.
            grid_nx, grid_ny = _resolve_grid_params(cfg, count)

            anchors = generate_anchors(
                area=area,
                pattern=pattern,
                count=count,
                margin_m=float(cfg["anchors"]["margin_m"]),
                radius_m=float(cfg["anchors"]["radius_m"]),
                grid_nx=grid_nx,
                grid_ny=grid_ny,
                seed=grid.base_seed,
            )

            for sigma in grid.sigmas:
                meas_cfg = {"range_noise_sigma_m": float(sigma)}

                cond = {
                    "pattern": pattern,
                    "requested_count": int(count),
                    "anchors_generated": int(anchors.shape[0]),
                    "grid_nx": int(grid_nx),
                    "grid_ny": int(grid_ny),
                    "sigma_m": float(sigma),
                    "trials": int(grid.trials),
                    "base_seed": int(grid.base_seed),
                    "dt": float(sim_time["dt"]),
                    "steps": int(sim_time["steps"]),
                }
                conditions_rows.append(cond)

                summary, _per_trial = run_monte_carlo(
                    area={"xmin": area.xmin, "xmax": area.xmax, "ymin": area.ymin, "ymax": area.ymax},
                    sim_time=sim_time,
                    traj_cfg=traj_cfg,
                    anchors=anchors,
                    meas_cfg=meas_cfg,
                    est_cfg=est_cfg,
                    base_seed=grid.base_seed,
                    trials=grid.trials,
                )

                # Merge condition fields + summary stats into one row
                row = {**cond, **summary}
                summary_rows.append(row)

                print(
                    f"[Batch] pattern={pattern:9s} reqN={count:2d} "
                    f"N={anchors.shape[0]:2d} sigma={sigma:.2f} "
                    f"rmse_mean={row['rmse_mean_m']:.3f}"
                )

    # Write CSVs
    pd.DataFrame(conditions_rows).to_csv(paths.conditions_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(paths.summary_csv, index=False)

    print(f"[Batch] Saved conditions: {paths.conditions_csv}")
    print(f"[Batch] Saved summary:    {paths.summary_csv}")


if __name__ == "__main__":
    main()
