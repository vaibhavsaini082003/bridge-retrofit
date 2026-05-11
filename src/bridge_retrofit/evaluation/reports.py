"""Evaluation reports for the full pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from bridge_retrofit.config import ProjectConfig
from bridge_retrofit.models.severity import train_severity_model
from bridge_retrofit.models.similarity import fit_similarity_index, recommend_retrofit_from_neighbors
from bridge_retrofit.preprocessing.pipeline import Preprocessors


def evaluate_all(
    cfg: ProjectConfig,
    df: pd.DataFrame,
    preprocessors: Preprocessors,
    feature_cols: list[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {}

    sev_model, sev_metrics = train_severity_model(cfg, df, preprocessors, feature_cols)
    report["severity"] = sev_metrics

    # Similarity-vote evaluation (simple holdout-style): fit on all rows then vote from neighbors.
    # This is optimistic but useful as a quick sanity check.
    index, case_table = fit_similarity_index(cfg, df, preprocessors, feature_cols)
    if cfg.columns.retrofit_target in df.columns:
        correct = 0
        total = 0
        X_sim = preprocessors.similarity_features.transform(df[feature_cols])
        for i in range(min(len(df), 200)):
            res = index.query(X_sim[i], k=cfg.similarity.vote_k + 1)
            # Skip self-match at distance 0
            neighbor_idx = [j for j in res.indices if j != i][: cfg.similarity.vote_k]
            rec = recommend_retrofit_from_neighbors(case_table.iloc[neighbor_idx], cfg.columns.retrofit_target)
            true = str(df.iloc[i][cfg.columns.retrofit_target])
            if rec is not None:
                correct += int(rec == true)
                total += 1
        report["similarity_vote"] = {
            "evaluated_rows": total,
            "top1_accuracy": (correct / total) if total else None,
        }

    return report
