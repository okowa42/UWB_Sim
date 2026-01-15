from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uwb_sim.io.config_loader import load_yaml_config
from uwb_sim.metrics.gdop import gdop_grid_2d, summarize_gdop
from uwb_sim.scenario.patterns import Area2D, generate_anchors


def _latest_file(glob_pattern: str) -> Path:
    files = list(Path(".").glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def _make_outdir(base: str = "outputs/gdop") -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = Path(base) / ts
    outdir.mkdir(parents=True, exist_ok=False)
    return outdir


def _resolve_grid_params(count: int, fallback_nx: int, fallback_ny: int) -> tuple[int, int]:
    root = int(round(count ** 0.5))
    if root * root == count and root >= 2:
        return root, root
    return fallback_nx, fallback_ny


def main():
    cfg = load_yaml_config(Path("configs/default.yaml"))
    area_cfg = cfg["simulation"]["area"]
    area = Area2D(float(area_cfg["xmin"]), float(area_cfg["xmax"]), float(area_cfg["ymin"]), float(area_cfg["ymax"]))
    area_dict = {"xmin": area.xmin, "xmax": area.xmax, "ymin": area.ymin, "ymax": area.ymax}

    # latest pareto
    pareto_csv = _latest_file("outputs/pareto/*/pareto.csv")
    pareto_df = pd.read_csv(pareto_csv)
    print(f"[GDOP] Using pareto: {pareto_csv}")

    outdir = _make_outdir()
    print(f"[GDOP] Output dir: {outdir}")

    # parameters
    grid_n = 61  # 61x61 grid
    margin_eval = 0.0

    margin_m = float(cfg["anchors"]["margin_m"])
    radius_m = float(cfg["anchors"]["radius_m"])
    base_seed = int(cfg["anchors"]["random"]["seed"])
    fallback_nx = int(cfg["anchors"]["grid"]["nx"])
    fallback_ny = int(cfg["anchors"]["grid"]["ny"])

    rows = []

    for idx, r in pareto_df.iterrows():
        pattern = str(r["pattern"])
        n = int(r["anchors_generated"])
        sigma = float(r.get("sigma_m", 0.0))

        grid_nx, grid_ny = _resolve_grid_params(n, fallback_nx, fallback_ny)

        anchors = generate_anchors(
            area=area,
            pattern=pattern,
            count=n,
            margin_m=margin_m,
            radius_m=radius_m,
            grid_nx=grid_nx,
            grid_ny=grid_ny,
            seed=base_seed,
        )

        xs, ys, Z = gdop_grid_2d(area_dict, anchors, grid_n=grid_n, margin=margin_eval)
        summ = summarize_gdop(Z)

        # save heatmap image
        fig, ax = plt.subplots()
        im = ax.imshow(
            Z,
            origin="lower",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            aspect="equal",
        )
        fig.colorbar(im, ax=ax, label="GDOP (2D)")
        ax.scatter(anchors[:, 0], anchors[:, 1], marker="s")
        ax.set_title(f"GDOP heatmap | pattern={pattern}, N={anchors.shape[0]}, sigma={sigma}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        img_path = outdir / f"gdop_{idx:02d}_{pattern}_N{anchors.shape[0]}_s{sigma:.2f}.png"
        fig.savefig(img_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        row = {
            "pattern": pattern,
            "anchors_generated": int(anchors.shape[0]),
            "sigma_m": sigma,
            "gdop_mean": summ["gdop_mean"],
            "gdop_p95": summ["gdop_p95"],
            "gdop_max": summ["gdop_max"],
            "gdop_grid_n": grid_n,
            "gdop_img": str(img_path),
        }

        # carry pareto metrics if present
        for col in ["rmse_mean_m", "wls_step_mean_s"]:
            if col in pareto_df.columns:
                row[col] = float(r[col])

        rows.append(row)

        print(f"[GDOP] {pattern:9s} N={anchors.shape[0]:2d} gdop_mean={row['gdop_mean']:.3f} saved={img_path.name}")

    out_csv = outdir / "gdop_pareto.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[GDOP] Saved summary: {out_csv}")


if __name__ == "__main__":
    main()
