from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import pandas as pd

from uwb_sim.io.paths import RunPaths


def export_summary_csv(paths: RunPaths, summary: Dict[str, Any]) -> None:
    """
    summary: dict (single row)
    """
    df = pd.DataFrame([summary])
    df.to_csv(paths.summary_csv, index=False)


def export_per_trial_csv(paths: RunPaths, per_trial_rows: list[dict[str, Any]]) -> None:
    """
    per_trial_rows: list of dicts (each trial row)
    """
    df = pd.DataFrame(per_trial_rows)
    df.to_csv(paths.per_trial_csv, index=False)
