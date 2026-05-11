"""Data ingestion utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bridge_retrofit.config import ProjectConfig


def load_dataset(cfg: ProjectConfig) -> pd.DataFrame:
    path = (cfg.project_root / cfg.data.raw_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=cfg.data.sheet_name)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset extension: {path.suffix}")

    df.columns = [str(c).strip() for c in df.columns]
    return df
