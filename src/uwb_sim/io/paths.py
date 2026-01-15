from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    summary_csv: Path
    per_trial_csv: Path


def make_run_paths(base_dir: str | Path = "outputs/runs") -> RunPaths:
    """
    Create a timestamped run directory, e.g. outputs/runs/2026-01-16_01-23-45
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=False)

    return RunPaths(
        run_dir=run_dir,
        summary_csv=run_dir / "summary.csv",
        per_trial_csv=run_dir / "per_trial.csv",
    )
