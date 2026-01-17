from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =========================
# Plot / filtering settings
# =========================

# Outlier handling for visualization (CSVは消さない。図からだけ除外)
# - "quantile": rmse_mean_m 上位 (1 - RMSE_QUANTILE) を除外
# - "threshold": rmse_mean_m > RMSE_THRESHOLD_M を除外
OUTLIER_MODE = "quantile"  # "quantile" or "threshold"
RMSE_QUANTILE = 0.99
RMSE_THRESHOLD_M = 1.0  # used only if OUTLIER_MODE == "threshold"

# Semi-log (y axis)
USE_LOG_Y = True
RMSE_EPS = 1e-6  # log用の下限

# Output base dir
OUT_BASEDIR = "outputs/pareto"

# Pattern -> marker mapping（直感的）
PATTERN_MARKERS = {
    "random": "*",     # 星
    "circle": "o",     # 丸
    "perimeter": "s",  # 四角
    # 追加パターンが来たときのフォールバック（必要ならここに追加）
    "grid": "P",
    "corners": "^",
}


# =========================
# Utilities
# =========================

def _latest_file(glob_pattern: str) -> Path:
    files = list(Path(".").glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def _make_outdir(base: str = OUT_BASEDIR) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outdir = Path(base) / ts
    outdir.mkdir(parents=True, exist_ok=False)
    return outdir


def _ensure_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    cols = {k: v for k, v in rename_map.items() if k in df.columns}
    if cols:
        df = df.rename(columns=cols)
    return df


def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()].copy()


def _pareto_minimize(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """
    Pareto front for minimizing both x_col and y_col.
    Efficient extraction:
      sort by x asc, keep points with strictly decreasing running min of y.
    """
    d = df.sort_values(by=[x_col, y_col], ascending=[True, True]).reset_index(drop=True)
    best_y = float("inf")
    keep = []
    for _, row in d.iterrows():
        y = float(row[y_col])
        if y < best_y - 1e-15:
            keep.append(True)
            best_y = y
        else:
            keep.append(False)
    return d.loc[keep].copy()


def _apply_outlier_filter_for_plot(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove extreme RMSE outliers ONLY for plotting readability.
    Returns (filtered_df, removed_count).
    """
    if df.empty:
        return df.copy(), 0

    if OUTLIER_MODE == "quantile":
        cut = float(df["rmse_mean_m"].quantile(RMSE_QUANTILE))
        before = len(df)
        df2 = df[df["rmse_mean_m"] <= cut].copy()
        return df2, before - len(df2)

    if OUTLIER_MODE == "threshold":
        before = len(df)
        df2 = df[df["rmse_mean_m"] <= float(RMSE_THRESHOLD_M)].copy()
        return df2, before - len(df2)

    return df.copy(), 0


def _compute_total_time_s(df: pd.DataFrame) -> pd.DataFrame:
    """
    研究テーマに合わせて「推定開始〜終了までの総計算時間」を作る。

    優先順位：
      1) wls_total_s がある -> それを採用（推定全体の総時間として記録されている前提）
      2) wls_step_mean_s と wls_steps_measured がある -> 積を採用
      3) wls_step_mean_s と steps がある -> 積を採用（trajectory steps を使う）
    """
    df = df.copy()

    if "wls_total_s" in df.columns and df["wls_total_s"].notna().any():
        df["total_time_s"] = df["wls_total_s"].astype(float)
        df["total_time_source"] = "wls_total_s"
        return df

    if "wls_step_mean_s" in df.columns:
        step_mean = df["wls_step_mean_s"].astype(float)

        if "wls_steps_measured" in df.columns and df["wls_steps_measured"].notna().any():
            n = df["wls_steps_measured"].fillna(0).astype(float)
            df["total_time_s"] = step_mean * n
            df["total_time_source"] = "wls_step_mean_s*wls_steps_measured"
            return df

        if "steps" in df.columns and df["steps"].notna().any():
            n = df["steps"].fillna(0).astype(float)
            df["total_time_s"] = step_mean * n
            df["total_time_source"] = "wls_step_mean_s*steps"
            return df

    raise ValueError(
        "Cannot compute total_time_s. Need one of:\n"
        "- wls_total_s\n"
        "- (wls_step_mean_s and wls_steps_measured)\n"
        "- (wls_step_mean_s and steps)\n"
        "Check your profile CSV and merged columns."
    )


# =========================
# Main
# =========================

def main():
    # 1) Find latest batch (accuracy) and profile (timing) CSVs
    batch_csv = _latest_file("outputs/batches/*/batch_summary.csv")
    prof_csv = _latest_file("outputs/profiles/*/batch_summary.csv")

    print(f"[Pareto] Using batch (accuracy): {batch_csv}")
    print(f"[Pareto] Using profile (timing): {prof_csv}")

    # 2) Load
    df_acc = pd.read_csv(batch_csv)
    df_tim = pd.read_csv(prof_csv)

    # 3) Normalize columns
    df_acc = _ensure_columns(df_acc, {
        "range_noise_sigma_m": "sigma_m",
        "anchors": "anchors_generated",
    })
    df_tim = _ensure_columns(df_tim, {
        "range_noise_sigma_m": "sigma_m",
        "anchors": "anchors_generated",
    })

    # 4) Drop duplicate labels defensively
    df_acc = _drop_duplicate_columns(df_acc)
    df_tim = _drop_duplicate_columns(df_tim)

    # 5) Validate required columns
    required_acc = {"pattern", "anchors_generated", "sigma_m", "rmse_mean_m"}
    required_tim = {"pattern", "anchors_generated", "sigma_m"}  # total timeは後で構成できる可能性あり

    missing_acc = sorted(list(required_acc - set(df_acc.columns)))
    missing_tim = sorted(list(required_tim - set(df_tim.columns)))

    if missing_acc:
        raise ValueError(f"Accuracy CSV missing columns: {missing_acc}")
    if missing_tim:
        raise ValueError(f"Timing CSV missing columns: {missing_tim}")

    # 6) Select columns for merge（衝突回避のため必要最小限＋使う列だけ）
    acc_cols = [
        "pattern", "anchors_generated", "sigma_m",
        "trials", "steps",  # total time fallbackに使えることがある
        "rmse_mean_m", "rmse_std_m",
        "mae_mean_m", "mae_std_m",
        "p95_mean_m", "p95_std_m",
    ]
    acc_cols = [c for c in acc_cols if c in df_acc.columns]
    df_acc_s = df_acc[acc_cols].copy()

    tim_cols = [
        "pattern", "anchors_generated", "sigma_m",
        "wls_total_s", "wls_step_mean_s", "wls_steps_measured",
    ]
    tim_cols = [c for c in tim_cols if c in df_tim.columns]
    df_tim_s = df_tim[tim_cols].copy()

    # 7) Merge
    key = ["pattern", "anchors_generated", "sigma_m"]
    merged = pd.merge(df_acc_s, df_tim_s, how="inner", on=key)

    if merged.empty:
        raise ValueError(
            "Merged result is empty. Likely mismatch in keys between accuracy and timing outputs.\n"
            "Check that both CSVs share: pattern, anchors_generated, sigma_m."
        )

    # 8) Compute total time (推定開始〜終了までの総時間)
    merged = _compute_total_time_s(merged)

    # 9) Pareto extraction (minimize total_time_s and rmse_mean_m)
    pareto = _pareto_minimize(merged, x_col="total_time_s", y_col="rmse_mean_m")

    # 10) Save CSVs
    outdir = _make_outdir()
    merged_path = outdir / "merged.csv"
    pareto_path = outdir / "pareto.csv"

    merged.to_csv(merged_path, index=False)
    pareto.to_csv(pareto_path, index=False)

    print(f"[Pareto] Saved merged: {merged_path}")
    print(f"[Pareto] Saved pareto: {pareto_path}")

    # 11) Plot per sigma (x=total_time_s, y=rmse_mean_m)
    sigmas = sorted(merged["sigma_m"].dropna().unique().tolist())
    removed_total = 0

    for sigma in sigmas:
        df_sigma = merged[merged["sigma_m"] == sigma].copy()
        if df_sigma.empty:
            continue

        # outlier filter for plotting readability (only for this sigma)
        plot_df, removed = _apply_outlier_filter_for_plot(df_sigma)
        removed_total += removed

        # log-safe RMSE
        plot_df = plot_df.copy()
        plot_df["rmse_plot_m"] = plot_df["rmse_mean_m"].clip(lower=RMSE_EPS)

        pareto_sigma = pareto[pareto["sigma_m"] == sigma].copy()
        pareto_sigma["rmse_plot_m"] = pareto_sigma["rmse_mean_m"].clip(lower=RMSE_EPS)

        fig, ax = plt.subplots()

        # color encodes anchor count
        # marker encodes pattern
        # Matplotlibは一度のscatterで markerを行ごとに変えられないので patternごとに描く
        all_patterns = sorted(plot_df["pattern"].dropna().unique().tolist())
        sc = None
        for pat in all_patterns:
            sub = plot_df[plot_df["pattern"] == pat]
            marker = PATTERN_MARKERS.get(str(pat), "x")
            sc = ax.scatter(
                sub["total_time_s"],
                sub["rmse_plot_m"],
                c=sub["anchors_generated"],
                marker=marker,
                alpha=0.65,
            )

        # Pareto candidates: subtle highlight (edge only)
        all_pareto_patterns = sorted(pareto_sigma["pattern"].dropna().unique().tolist())
        for pat in all_pareto_patterns:
            sub = pareto_sigma[pareto_sigma["pattern"] == pat]
            marker = PATTERN_MARKERS.get(str(pat), "x")
            ax.scatter(
                sub["total_time_s"],
                sub["rmse_plot_m"],
                c=sub["anchors_generated"],
                marker=marker,
                edgecolors="black",
                linewidths=1.2,
                alpha=1.0,
            )

        ax.set_xlabel("Total WLS computation time per run [s]")
        ax.set_ylabel("RMSE mean [m]")

        if USE_LOG_Y:
            ax.set_yscale("log")
            ax.set_title(f"Accuracy vs Total Computation (sigma={sigma:g}, y log)")
        else:
            ax.set_title(f"Accuracy vs Total Computation (sigma={sigma:g})")

        # colorbar for anchor count（scがNoneになることは通常ないが保険）
        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("Anchor count (anchors_generated)")

        # pattern legend (marker explanation)
        handles = []
        labels = []
        for pat in sorted(set(all_patterns) | set(all_pareto_patterns)):
            marker = PATTERN_MARKERS.get(str(pat), "x")
            h = ax.scatter([], [], marker=marker, c="gray")
            handles.append(h)
            labels.append(str(pat))
        if handles:
            ax.legend(handles, labels, title="Pattern (marker)", loc="upper right")

        # show only removed count
        if removed > 0:
            ax.text(
                0.01,
                0.01,
                f"Outliers removed for plot: {removed}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="bottom",
            )

        scatter_path = outdir / f"scatter_total_sigma_{sigma:g}.png"
        fig.savefig(scatter_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"[Pareto] Saved plot (sigma={sigma:g}): {scatter_path}")

    # 12) Console summary
    print("\n[Pareto] Pareto candidates (sorted by total time):")
    show_cols = [
        "pattern", "anchors_generated", "sigma_m",
        "rmse_mean_m", "total_time_s", "total_time_source",
        "wls_step_mean_s", "wls_steps_measured", "wls_total_s",
    ]
    show_cols = [c for c in show_cols if c in pareto.columns]
    print(pareto.sort_values("total_time_s")[show_cols].to_string(index=False))

    if removed_total > 0:
        print(f"\n[Pareto] Note: total outliers removed for plots: {removed_total}")


if __name__ == "__main__":
    main()
