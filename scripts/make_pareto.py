from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# ---------- utilities ----------

def _latest_file(glob_pattern: str) -> Path:
    files = list(Path(".").glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")
    # directory timestamps are in the parent folder name; but we use mtime as robust fallback
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def _ensure_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Rename columns if present."""
    cols = {k: v for k, v in rename_map.items() if k in df.columns}
    if cols:
        df = df.rename(columns=cols)
    return df

def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas allows duplicate column labels; merge fails on non-unique keys.
    Keep the first occurrence and drop the rest.
    """
    return df.loc[:, ~df.columns.duplicated()].copy()



def _make_outdir(base: str = "outputs/pareto") -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = Path(base) / ts
    outdir.mkdir(parents=True, exist_ok=False)
    return outdir


def _pareto_minimize(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """
    Pareto front for minimizing both x_col and y_col.

    Condition:
      a dominates b if a.x <= b.x and a.y <= b.y and at least one strict.
    """
    # Sort by x ascending; keep points with strictly decreasing y minima.
    d = df.sort_values(by=[x_col, y_col], ascending=[True, True]).reset_index(drop=True)
    best_y = float("inf")
    keep = []
    for i, row in d.iterrows():
        y = float(row[y_col])
        if y < best_y - 1e-15:
            keep.append(True)
            best_y = y
        else:
            keep.append(False)
    return d.loc[keep].copy()


# ---------- main logic ----------

def main():
    # 1) Find latest batch (MC accuracy) and profile (timing) CSVs
    # Step2 output: outputs/batches/<ts>/batch_summary.csv
    # Step4 output: outputs/profiles/<ts>/batch_summary.csv  (or similar)
    batch_csv = _latest_file("outputs/batches/*/batch_summary.csv")
    prof_csv = _latest_file("outputs/profiles/*/batch_summary.csv")

    print(f"[Pareto] Using batch (accuracy): {batch_csv}")
    print(f"[Pareto] Using profile (timing): {prof_csv}")

    # 2) Load
    df_acc = pd.read_csv(batch_csv)
    df_tim = pd.read_csv(prof_csv)

    # 3) Normalize column names (absorb variations)
    # Accuracy side: range_noise_sigma_m vs sigma_m, anchors_generated etc.
    df_acc = _ensure_columns(df_acc, {
        "range_noise_sigma_m": "sigma_m",
        "anchors": "anchors_generated",  # if any older export used "anchors"
    })
    # Timing side: already sigma_m typically, but keep compatibility
    df_tim = _ensure_columns(df_tim, {
        "range_noise_sigma_m": "sigma_m",
        "anchors": "anchors_generated",
    })
    df_acc = _drop_duplicate_columns(df_acc)
    df_tim = _drop_duplicate_columns(df_tim)


    # 4) Validate required columns
    required_acc = {"pattern", "anchors_generated", "sigma_m", "rmse_mean_m"}
    required_tim = {"pattern", "anchors_generated", "sigma_m", "wls_step_mean_s"}

    missing_acc = sorted(list(required_acc - set(df_acc.columns)))
    missing_tim = sorted(list(required_tim - set(df_tim.columns)))

    if missing_acc:
        raise ValueError(f"Accuracy CSV missing columns: {missing_acc}")
    if missing_tim:
        raise ValueError(f"Timing CSV missing columns: {missing_tim}")

    # 5) Select only the necessary columns for merge (avoid collisions)
    acc_cols = [
        "pattern", "anchors_generated", "sigma_m",
        "trials", "rmse_mean_m", "rmse_std_m", "mae_mean_m", "mae_std_m", "p95_mean_m", "p95_std_m"
    ]
    acc_cols = [c for c in acc_cols if c in df_acc.columns]
    df_acc_s = df_acc[acc_cols].copy()

    tim_cols = ["pattern", "anchors_generated", "sigma_m", "wls_step_mean_s", "wls_total_s", "wls_steps_measured"]
    tim_cols = [c for c in tim_cols if c in df_tim.columns]
    df_tim_s = df_tim[tim_cols].copy()

    # 6) Merge
    key = ["pattern", "anchors_generated", "sigma_m"]
    for name, df in [("accuracy", df_acc), ("timing", df_tim)]:
        dups = df.columns[df.columns.duplicated()].tolist()
        if dups:
            print(f"[Pareto] Warning: {name} had duplicate columns, dropped: {dups}")

    merged = pd.merge(df_acc_s, df_tim_s, how="inner", on=key)

    if merged.empty:
        raise ValueError(
            "Merged result is empty. Likely mismatch in keys between Step2 and Step4 outputs.\n"
            "Check that both CSVs share: pattern, anchors_generated, sigma_m."
        )

    # 7) Pareto extraction (minimize rmse_mean_m and wls_step_mean_s)
    pareto = _pareto_minimize(merged, x_col="wls_step_mean_s", y_col="rmse_mean_m")

    # 8) Save
    outdir = _make_outdir()
    merged_path = outdir / "merged.csv"
    pareto_path = outdir / "pareto.csv"

    merged.to_csv(merged_path, index=False)
    pareto.to_csv(pareto_path, index=False)

    print(f"[Pareto] Saved merged: {merged_path}")
    print(f"[Pareto] Saved pareto: {pareto_path}")

    # 9) Plot (scatter + pareto)
    fig, ax = plt.subplots()
    ax.scatter(merged["wls_step_mean_s"], merged["rmse_mean_m"])
    ax.scatter(pareto["wls_step_mean_s"], pareto["rmse_mean_m"])
    ax.set_xlabel("WLS step mean time [s]")
    ax.set_ylabel("RMSE mean [m]")
    ax.set_title("Accuracy vs Computation (Pareto highlighted)")
    scatter_path = outdir / "scatter.png"
    fig.savefig(scatter_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[Pareto] Saved plot: {scatter_path}")

    # 10) Quick console summary
    print("\n[Pareto] Pareto candidates (sorted by time):")
    show_cols = ["pattern", "anchors_generated", "sigma_m", "rmse_mean_m", "wls_step_mean_s"]
    show_cols = [c for c in show_cols if c in pareto.columns]
    print(pareto.sort_values("wls_step_mean_s")[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
