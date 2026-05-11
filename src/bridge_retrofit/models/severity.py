"""Severity model training."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from bridge_retrofit.config import ProjectConfig
from bridge_retrofit.preprocessing.pipeline import Preprocessors


def _make_estimator(cfg: ProjectConfig, name: str):
    name = (name or "auto").lower()
    if name == "auto":
        try:
            import lightgbm as lgb  # type: ignore

            return lgb.LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                random_state=cfg.training.random_seed,
                class_weight="balanced",
                n_jobs=-1,
            )
        except Exception:
            name = "logreg"

    if name == "lgbm":
        import lightgbm as lgb  # type: ignore

        return lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            random_state=cfg.training.random_seed,
            class_weight="balanced",
            n_jobs=-1,
        )

    if name in {"xgb", "xgboost"}:
        try:
            import xgboost as xgb  # type: ignore
        except Exception as e:
            raise RuntimeError("xgboost is not installed; install it (Colab: pip install -r requirements-colab.txt)") from e

        device = (cfg.training.device or "auto").lower()
        params: dict[str, Any] = {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": cfg.training.random_seed,
            "n_jobs": -1,
            "eval_metric": "mlogloss",
        }

        # GPU is optional; keep CPU-safe defaults.
        if device == "gpu":
            # XGBoost 2.x prefers device='cuda'. Older setups use tree_method='gpu_hist'.
            params["tree_method"] = "hist"
            params["device"] = "cuda"

        return xgb.XGBClassifier(**params)

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            class_weight="balanced",
            random_state=cfg.training.random_seed,
            n_jobs=-1,
        )

    if name == "logreg":
        return LogisticRegression(
            max_iter=5000,
            n_jobs=-1,
            class_weight="balanced",
            solver="saga",
            multi_class="auto",
        )

    raise ValueError(f"Unknown severity_model: {name}")


def train_severity_model(
    cfg: ProjectConfig,
    df: pd.DataFrame,
    preprocessors: Preprocessors,
    feature_cols: list[str],
) -> tuple[Any, dict[str, Any]]:
    target = cfg.columns.severity_target
    if target not in df.columns:
        raise KeyError(f"Severity target column not found: {target}")

    X = preprocessors.model_features.transform(df[feature_cols])
    y = df[target].astype(str).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    est = _make_estimator(cfg, cfg.training.severity_model)
    est.fit(X_train, y_train)

    y_pred = est.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics: dict[str, Any] = {
        "task": "severity",
        "n_train": int(getattr(X_train, "shape", [0])[0]),
        "n_test": int(getattr(X_test, "shape", [0])[0]),
        "classes": sorted(list(set(map(str, y)))),
        "classification_report": report,
    }
    return est, metrics
