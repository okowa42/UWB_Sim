from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """
    Load YAML config file and return as dict.

    Notes:
      - This project intentionally keeps config handling simple at Phase 1.
      - Validation (pydantic) can be added later without changing call sites.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Invalid config: root must be a mapping (dict).")

    return data
