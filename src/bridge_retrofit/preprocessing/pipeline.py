"""Preprocessing pipelines for modeling and similarity retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bridge_retrofit.config import ProjectConfig


@dataclass
class Preprocessors:
    model_features: Pipeline
    similarity_features: Pipeline
    numeric_features: list[str]
    categorical_features: list[str]


def infer_feature_columns(cfg: ProjectConfig, df: pd.DataFrame) -> tuple[list[str], list[str]]:
    severity = cfg.columns.severity_target
    retrofit = cfg.columns.retrofit_target
    target_cols = [c for c in [severity, retrofit] if c in df.columns]

    excluded = set(cfg.columns.exclude_features + target_cols)
    feature_cols = [c for c in df.columns if c not in excluded]
    return feature_cols, target_cols


def _split_numeric_categorical(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    numeric = []
    categorical = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical


def fit_preprocessors(cfg: ProjectConfig, df: pd.DataFrame, feature_cols: list[str]) -> Preprocessors:
    num_cols, cat_cols = _split_numeric_categorical(df, feature_cols)

    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=cfg.preprocessing.onehot_min_frequency,
                    max_categories=cfg.preprocessing.onehot_max_categories,
                    sparse_output=True,
                ),
            ),
        ]
    )

    base_features = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model_features = Pipeline(steps=[("features", base_features)])
    model_features.fit(df[feature_cols])

    # Similarity pipeline: same encoding, then (optional) TruncatedSVD to get a compact embedding.
    sim_features = clone(base_features)
    sim_steps: list[tuple[str, Any]] = [("features", sim_features)]
    if cfg.preprocessing.svd_components and cfg.preprocessing.svd_components > 0:
        sim_steps.append(("svd", TruncatedSVD(n_components=cfg.preprocessing.svd_components, random_state=cfg.training.random_seed)))
        sim_steps.append(("scale", StandardScaler(with_mean=True)))

    similarity_features = Pipeline(steps=sim_steps)
    similarity_features.fit(df[feature_cols])

    return Preprocessors(
        model_features=model_features,
        similarity_features=similarity_features,
        numeric_features=num_cols,
        categorical_features=cat_cols,
    )


def make_single_row_frame(payload: dict[str, Any], feature_cols: list[str]) -> pd.DataFrame:
    row = {c: payload.get(c, np.nan) for c in feature_cols}
    return pd.DataFrame([row])



def coerce_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): coerce_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [coerce_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj
