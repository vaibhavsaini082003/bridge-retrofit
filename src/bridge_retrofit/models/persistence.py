"""Model persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from bridge_retrofit.config import ProjectConfig


def artifact_dir(cfg: ProjectConfig, run_id: str) -> Path:
    out = (cfg.project_root / cfg.artifacts.out_dir / run_id)
    out.mkdir(parents=True, exist_ok=True)
    return out


def list_run_dirs(cfg: ProjectConfig) -> list[Path]:
    root = (cfg.project_root / cfg.artifacts.out_dir)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], reverse=True)


def latest_run_dir_with(cfg: ProjectConfig, required_files: list[str]) -> Path | None:
    for run_dir in list_run_dirs(cfg):
        if all((run_dir / f).exists() for f in required_files):
            return run_dir
    return None


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def save_artifact(run_dir: Path, name: str, obj: Any) -> Path:
    path = run_dir / name
    if name.endswith(".json") and not isinstance(obj, (str, bytes)):
        save_json(path, obj)
        return path
    if name.endswith(".joblib"):
        joblib.dump(obj, path)
        return path
    # default: json
    save_json(path, obj)
    return path


def load_artifact(run_dir: Path, name: str) -> Any:
    path = run_dir / name
    if name.endswith(".joblib"):
        return joblib.load(path)
    if name.endswith(".json"):
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unknown artifact type: {name}")
