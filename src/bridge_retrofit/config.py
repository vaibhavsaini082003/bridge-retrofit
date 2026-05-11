"""Configuration loading.

Config is stored as YAML to keep Colab usage simple.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    raw_path: str
    sheet_name: str | None
    processed_dir: str


@dataclass(frozen=True)
class ColumnsConfig:
    severity_target: str
    retrofit_target: str
    exclude_features: list[str]
    case_display: list[str]


@dataclass(frozen=True)
class PreprocessingConfig:
    svd_components: int
    onehot_min_frequency: int
    onehot_max_categories: int | None


@dataclass(frozen=True)
class TrainingConfig:
    random_seed: int
    test_size: float
    severity_model: str
    retrofit_model: str
    device: str


@dataclass(frozen=True)
class SimilarityConfig:
    backend: str
    n_neighbors: int
    vote_k: int


@dataclass(frozen=True)
class ArtifactsConfig:
    out_dir: str


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    data: DataConfig
    columns: ColumnsConfig
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    similarity: SimilarityConfig
    artifacts: ArtifactsConfig


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required config key: {key}")
    return d[key]


def load_config(config_path: Path, project_root_override: str | None = None) -> ProjectConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping")

    project_root = Path(project_root_override or _require(raw, "project_root")).expanduser()
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()

    data = _require(raw, "data")
    cols = _require(raw, "columns")

    preprocessing = raw.get("preprocessing", {})
    training = raw.get("training", {})
    similarity = raw.get("similarity", {})
    artifacts = raw.get("artifacts", {})

    return ProjectConfig(
        project_root=project_root,
        data=DataConfig(
            raw_path=str(_require(data, "raw_path")),
            sheet_name=data.get("sheet_name"),
            processed_dir=str(_require(data, "processed_dir")),
        ),
        columns=ColumnsConfig(
            severity_target=str(_require(cols, "severity_target")),
            retrofit_target=str(_require(cols, "retrofit_target")),
            exclude_features=list(cols.get("exclude_features", [])),
            case_display=list(cols.get("case_display", [])),
        ),
        preprocessing=PreprocessingConfig(
            svd_components=int(preprocessing.get("svd_components", 0)),
            onehot_min_frequency=int(preprocessing.get("onehot_min_frequency", 1)),
            onehot_max_categories=(
                int(preprocessing["onehot_max_categories"]) if preprocessing.get("onehot_max_categories") is not None else None
            ),
        ),
        training=TrainingConfig(
            random_seed=int(training.get("random_seed", 42)),
            test_size=float(training.get("test_size", 0.2)),
            severity_model=str(training.get("severity_model", "auto")),
            retrofit_model=str(training.get("retrofit_model", "off")),
            device=str(training.get("device", "auto")),
        ),
        similarity=SimilarityConfig(
            backend=str(similarity.get("backend", "sklearn")),
            n_neighbors=int(similarity.get("n_neighbors", 6)),
            vote_k=int(similarity.get("vote_k", 5)),
        ),
        artifacts=ArtifactsConfig(out_dir=str(artifacts.get("out_dir", "artifacts"))),
    )
