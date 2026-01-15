from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from uwb_sim.io.config_loader import load_yaml_config
from uwb_sim.io.exporters import export_per_trial_csv, export_summary_csv
from uwb_sim.io.paths import make_run_paths
from uwb_sim.scenario.patterns import Area2D, generate_anchors
from uwb_sim.simulation.runner import run_monte_carlo


def _plot_area_and_anchors(ax, area: Area2D, anchors: np.ndarray, show_labels: bool = True):
    xs = [area.xmin, area.xmax, area.xmax, area.xmin, area.xmin]
    ys = [area.ymin, area.ymin, area.ymax, area.ymax, area.ymin]
    ax.plot(xs, ys)
    ax.scatter(anchors[:, 0], anchors[:, 1], marker="s")
    if show_labels:
        for i, (x, y) in enumerate(anchors):
            ax.text(x, y, f"A{i}", fontsize=10, ha="left", va="bottom")


def main():
    st.set_page_config(page_title="UWB 2D Localization Simulator", layout="wide")
    st.title("UWB 2D Localization Simulator (Phase 1: MC + WLS + LOS noise)")

    cfg = load_yaml_config(Path("configs/default.yaml"))

    sim = cfg["simulation"]
    area_cfg = sim["area"]
    area = Area2D(
        xmin=float(area_cfg["xmin"]),
        xmax=float(area_cfg["xmax"]),
        ymin=float(area_cfg["ymin"]),
        ymax=float(area_cfg["ymax"]),
    )

    # Sidebar: anchors
    st.sidebar.header("Anchor Settings")
    pattern = st.sidebar.selectbox(
        "Pattern",
        options=["corners", "perimeter", "circle", "grid", "random"],
        index=["corners", "perimeter", "circle", "grid", "random"].index(cfg["anchors"]["pattern"]),
    )
    count = int(st.sidebar.slider("Count (perimeter/circle/random)", 1, 20, int(cfg["anchors"]["count"])))
    margin_m = float(st.sidebar.slider("Margin [m]", 0.0, 20.0, float(cfg["anchors"]["margin_m"])))
    radius_m = float(st.sidebar.slider("Circle radius [m]", 5.0, 80.0, float(cfg["anchors"]["radius_m"])))
    grid_nx = int(st.sidebar.slider("Grid nx", 2, 10, int(cfg["anchors"]["grid"]["nx"])))
    grid_ny = int(st.sidebar.slider("Grid ny", 2, 10, int(cfg["anchors"]["grid"]["ny"])))
    base_seed = int(st.sidebar.number_input("Base seed", value=int(cfg["anchors"]["random"]["seed"]), step=1))
    show_labels = bool(st.sidebar.checkbox("Show anchor labels", value=bool(cfg["ui"]["show_anchor_labels"])))

    anchors = generate_anchors(
        area=area,
        pattern=pattern,
        count=count,
        margin_m=margin_m,
        radius_m=radius_m,
        grid_nx=grid_nx,
        grid_ny=grid_ny,
        seed=base_seed,
    )

    # Sidebar: simulation
    st.sidebar.header("Simulation Settings")
    dt = float(st.sidebar.slider("dt [s]", 0.1, 2.0, float(sim["time"]["dt"])))
    steps = int(st.sidebar.slider("steps", 50, 2000, int(sim["time"]["steps"])))
    sigma_m = float(st.sidebar.slider("range noise sigma [m]", 0.0, 1.0, float(cfg["measurement"]["range_noise_sigma_m"])))
    trials = int(st.sidebar.slider("Monte Carlo trials", 1, 200, 30))

    run_btn = st.sidebar.button("Run Monte Carlo")
    save_btn = st.sidebar.button("Save results to outputs/")

    # Layout
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots()
        _plot_area_and_anchors(ax, area, anchors, show_labels=show_labels)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Area & Anchors")
        ax.set_xlim(area.xmin - 5, area.xmax + 5)
        ax.set_ylim(area.ymin - 5, area.ymax + 5)
        st.pyplot(fig, clear_figure=True)

    with col2:
        st.subheader("Anchor Coordinates [m]")
        st.dataframe(
            {"id": [f"A{i}" for i in range(anchors.shape[0])], "x": anchors[:, 0], "y": anchors[:, 1]},
            use_container_width=True,
        )

    # Run MC
    if run_btn:
        area_dict = {"xmin": area.xmin, "xmax": area.xmax, "ymin": area.ymin, "ymax": area.ymax}
        sim_time = {"dt": dt, "steps": steps}
        traj_cfg = cfg["trajectory"]
        meas_cfg = {"range_noise_sigma_m": sigma_m}
        est_cfg = cfg["estimation"]

        summary, per_trial = run_monte_carlo(
            area=area_dict,
            sim_time=sim_time,
            traj_cfg=traj_cfg,
            anchors=anchors,
            meas_cfg=meas_cfg,
            est_cfg=est_cfg,
            base_seed=base_seed,
            trials=trials,
        )

        st.session_state["last_summary"] = summary
        st.session_state["last_per_trial"] = per_trial

    # Display results if available
    if "last_summary" in st.session_state:
        st.subheader("Monte Carlo Summary")
        st.write(st.session_state["last_summary"])

        st.subheader("Per-trial Metrics")
        df = pd.DataFrame(st.session_state["last_per_trial"])
        st.dataframe(df, use_container_width=True)

        # quick plot: RMSE distribution
        fig2, ax2 = plt.subplots()
        ax2.hist(df["rmse_m"].values, bins=20)
        ax2.set_xlabel("RMSE [m]")
        ax2.set_ylabel("count")
        ax2.set_title("RMSE distribution (MC)")
        st.pyplot(fig2, clear_figure=True)

    # Save
    if save_btn:
        if "last_summary" not in st.session_state:
            st.warning("No results to save. Run Monte Carlo first.")
        else:
            paths = make_run_paths()
            export_summary_csv(paths, st.session_state["last_summary"])
            export_per_trial_csv(paths, st.session_state["last_per_trial"])
            st.success(f"Saved to: {paths.run_dir}")


if __name__ == "__main__":
    main()
