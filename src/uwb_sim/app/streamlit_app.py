from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from uwb_sim.io.config_loader import load_yaml_config
from uwb_sim.scenario.patterns import Area2D, generate_anchors


def _plot_area_and_anchors(area: Area2D, anchors: np.ndarray, show_labels: bool = True):
    fig, ax = plt.subplots()

    # Area boundary
    xs = [area.xmin, area.xmax, area.xmax, area.xmin, area.xmin]
    ys = [area.ymin, area.ymin, area.ymax, area.ymax, area.ymin]
    ax.plot(xs, ys)

    # Anchors
    ax.scatter(anchors[:, 0], anchors[:, 1], marker="s")
    if show_labels:
        for i, (x, y) in enumerate(anchors):
            ax.text(x, y, f"A{i}", fontsize=10, ha="left", va="bottom")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Anchor Layout (2D, 100m x 100m)")

    ax.set_xlim(area.xmin - 5, area.xmax + 5)
    ax.set_ylim(area.ymin - 5, area.ymax + 5)

    return fig


def main():
    st.set_page_config(page_title="UWB 2D Localization Simulator", layout="wide")
    st.title("UWB 2D Localization Simulator (Phase 1: Anchor Layout UI)")

    # Load config
    config_path = Path("configs/default.yaml")
    cfg = load_yaml_config(config_path)

    sim = cfg["simulation"]
    area_cfg = sim["area"]
    area = Area2D(
        xmin=float(area_cfg["xmin"]),
        xmax=float(area_cfg["xmax"]),
        ymin=float(area_cfg["ymin"]),
        ymax=float(area_cfg["ymax"]),
    )

    # Sidebar controls
    st.sidebar.header("Anchor Settings")

    pattern = st.sidebar.selectbox(
        "Pattern",
        options=["corners", "perimeter", "circle", "grid", "random"],
        index=["corners", "perimeter", "circle", "grid", "random"].index(
            cfg["anchors"]["pattern"]
        )
        if "anchors" in cfg and "pattern" in cfg["anchors"]
        else 0,
    )

    count = int(st.sidebar.slider("Count (used by perimeter/circle/random)", 1, 20, int(cfg["anchors"]["count"])))
    margin_m = float(st.sidebar.slider("Margin [m]", 0.0, 20.0, float(cfg["anchors"]["margin_m"])))
    radius_m = float(st.sidebar.slider("Circle radius [m]", 5.0, 80.0, float(cfg["anchors"]["radius_m"])))

    grid_nx = int(st.sidebar.slider("Grid nx", 2, 10, int(cfg["anchors"]["grid"]["nx"])))
    grid_ny = int(st.sidebar.slider("Grid ny", 2, 10, int(cfg["anchors"]["grid"]["ny"])))

    seed = int(st.sidebar.number_input("Random seed", value=int(cfg["anchors"]["random"]["seed"]), step=1))
    show_labels = bool(st.sidebar.checkbox("Show anchor labels", value=bool(cfg["ui"]["show_anchor_labels"])))

    # Generate anchors
    anchors = generate_anchors(
        area=area,
        pattern=pattern,
        count=count,
        margin_m=margin_m,
        radius_m=radius_m,
        grid_nx=grid_nx,
        grid_ny=grid_ny,
        seed=seed,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = _plot_area_and_anchors(area, anchors, show_labels=show_labels)
        st.pyplot(fig, clear_figure=True)

    with col2:
        st.subheader("Current Parameters")
        st.write(
            {
                "area": {"xmin": area.xmin, "xmax": area.xmax, "ymin": area.ymin, "ymax": area.ymax},
                "pattern": pattern,
                "count": int(count),
                "margin_m": float(margin_m),
                "radius_m": float(radius_m),
                "grid": {"nx": int(grid_nx), "ny": int(grid_ny)},
                "random_seed": int(seed),
                "anchors_generated": int(anchors.shape[0]),
            }
        )

        st.subheader("Anchor Coordinates [m]")
        st.dataframe(
            {
                "id": [f"A{i}" for i in range(anchors.shape[0])],
                "x": anchors[:, 0],
                "y": anchors[:, 1],
            },
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
