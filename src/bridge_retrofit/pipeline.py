"""High-level pipeline orchestration.

The CLI calls into this module; individual steps live in focused modules.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bridge_retrofit.config import ProjectConfig
from bridge_retrofit.data.io import load_dataset
from bridge_retrofit.evaluation.reports import evaluate_all
from bridge_retrofit.models.persistence import (
    artifact_dir,
    load_artifact,
    latest_run_dir_with,
    save_artifact,
    save_json,
)
from bridge_retrofit.models.retrofit import train_retrofit_model
from bridge_retrofit.models.severity import train_severity_model
from bridge_retrofit.models.similarity import (
    fit_similarity_index,
    recommend_retrofit_from_neighbors,
)
from bridge_retrofit.preprocessing.pipeline import (
    fit_preprocessors,
    infer_feature_columns,
    make_single_row_frame,
)


def _run_id() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_preprocess(cfg: ProjectConfig) -> Path:
    run_dir = artifact_dir(cfg, run_id=_run_id())
    df = load_dataset(cfg)

    feature_cols, target_cols = infer_feature_columns(cfg, df)
    preprocessors = fit_preprocessors(cfg, df, feature_cols)

    out_processed = (cfg.project_root / cfg.data.processed_dir)
    out_processed.mkdir(parents=True, exist_ok=True)
    df_out = df[feature_cols + target_cols].copy()
    df_out.to_csv(out_processed / "dataset_selected.csv", index=False)

    save_artifact(run_dir, "feature_cols.json", feature_cols)
    save_artifact(run_dir, "target_cols.json", target_cols)
    save_artifact(run_dir, "preprocessors.joblib", preprocessors)

    save_json(run_dir / "run_metadata.json", {"step": "preprocess", "config": asdict(cfg)})
    return run_dir


def run_train(cfg: ProjectConfig, task: str) -> Path:
    run_dir = artifact_dir(cfg, run_id=_run_id())
    df = load_dataset(cfg)
    feature_cols, _ = infer_feature_columns(cfg, df)
    preprocessors = fit_preprocessors(cfg, df, feature_cols)

    if task == "severity":
        model, metrics = train_severity_model(cfg, df, preprocessors, feature_cols)
        save_artifact(run_dir, "severity_model.joblib", model)
        save_artifact(run_dir, "severity_metrics.json", metrics)
    elif task == "retrofit":
        model, metrics = train_retrofit_model(cfg, df, preprocessors, feature_cols)
        save_artifact(run_dir, "retrofit_model.joblib", model)
        save_artifact(run_dir, "retrofit_metrics.json", metrics)
    else:
        raise ValueError("task must be 'severity' or 'retrofit'")

    save_artifact(run_dir, "preprocessors.joblib", preprocessors)
    save_artifact(run_dir, "feature_cols.json", feature_cols)
    save_json(run_dir / "run_metadata.json", {"step": f"train:{task}", "timestamp": run_dir.name})
    return run_dir


def run_fit_similarity(cfg: ProjectConfig) -> Path:
    run_dir = artifact_dir(cfg, run_id=_run_id())
    df = load_dataset(cfg)
    feature_cols, _ = infer_feature_columns(cfg, df)
    preprocessors = fit_preprocessors(cfg, df, feature_cols)

    index, case_table = fit_similarity_index(cfg, df, preprocessors, feature_cols)
    save_artifact(run_dir, "preprocessors.joblib", preprocessors)
    save_artifact(run_dir, "similarity_index.joblib", index)
    case_table.to_csv(run_dir / "case_table.csv", index=False)
    return run_dir


def run_evaluate(cfg: ProjectConfig) -> dict[str, Any]:
    df = load_dataset(cfg)
    feature_cols, _ = infer_feature_columns(cfg, df)
    preprocessors = fit_preprocessors(cfg, df, feature_cols)
    report = evaluate_all(cfg, df, preprocessors, feature_cols)

    run_dir = artifact_dir(cfg, run_id=_run_id())
    save_artifact(run_dir, "evaluation_report.json", report)
    return report


def run_predict(cfg: ProjectConfig, payload: dict[str, Any]) -> dict[str, Any]:
    # Prediction loads the latest saved artifacts (timestamped run folders) if present.
    # For the "full pipeline" case we can combine the latest severity model and the
    # latest similarity index even if they were created in different runs.
    df = load_dataset(cfg)
    feature_cols, _ = infer_feature_columns(cfg, df)

    severity_run = latest_run_dir_with(cfg, ["preprocessors.joblib", "severity_model.joblib"])
    similarity_run = latest_run_dir_with(cfg, ["preprocessors.joblib", "similarity_index.joblib", "case_table.csv"])

    severity_preprocessors = load_artifact(severity_run, "preprocessors.joblib") if severity_run else None
    severity_model = load_artifact(severity_run, "severity_model.joblib") if severity_run else None

    similarity_preprocessors = load_artifact(similarity_run, "preprocessors.joblib") if similarity_run else None
    similarity_index = load_artifact(similarity_run, "similarity_index.joblib") if similarity_run else None
    case_table = pd.read_csv(similarity_run / "case_table.csv") if similarity_run else None

    # Fallback: if nothing exists yet, fit preprocessors on the fly so at least schema
    # handling works (but predictions will be empty until models are trained).
    if severity_preprocessors is None and similarity_preprocessors is None:
        onfly = fit_preprocessors(cfg, df, feature_cols)
        severity_preprocessors = onfly
        similarity_preprocessors = onfly

    row_df = make_single_row_frame(payload, feature_cols)
    result: dict[str, Any] = {}

    if severity_model is not None:
        x_model = severity_preprocessors.model_features.transform(row_df)
        result["severity_pred"] = severity_model.predict(x_model).tolist()[0]
        if hasattr(severity_model, "predict_proba"):
            proba = severity_model.predict_proba(x_model)
            result["severity_proba"] = proba.tolist()[0]

    if similarity_index is not None and case_table is not None:
        x_sim = similarity_preprocessors.similarity_features.transform(row_df)
        neighbors = similarity_index.query(x_sim, k=cfg.similarity.vote_k)
        result["similar_cases"] = case_table.iloc[neighbors.indices].assign(distance=neighbors.distances).to_dict(orient="records")
        result["recommended_retrofit"] = recommend_retrofit_from_neighbors(case_table.iloc[neighbors.indices], cfg.columns.retrofit_target)
    return result
