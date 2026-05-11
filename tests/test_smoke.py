from __future__ import annotations

import pandas as pd

from bridge_retrofit.config import load_config
from bridge_retrofit.preprocessing.pipeline import fit_preprocessors, infer_feature_columns


def test_preprocessor_fit(tmp_path):
    cfg_text = """
project_root: "."
data: {raw_path: "data/raw/bridge_data.xlsx", sheet_name: null, processed_dir: "data/processed"}
columns:
  severity_target: "Severity"
  retrofit_target: "Suggested_Retrofit"
  exclude_features: []
  case_display: []
preprocessing: {svd_components: 5, onehot_min_frequency: 1}
training: {random_seed: 42, test_size: 0.2, severity_model: "logreg", retrofit_model: "off"}
similarity: {n_neighbors: 3, vote_k: 2}
artifacts: {out_dir: "artifacts"}
"""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(cfg_text, encoding="utf-8")
    cfg = load_config(cfg_file)

    df = pd.DataFrame(
        {
            "Bridge_Type": ["Beam", "Arch", "Beam"],
            "Age_Years": [10, 50, 30],
            "Severity": ["Low", "High", "Medium"],
            "Suggested_Retrofit": ["A", "B", "A"],
        }
    )
    feature_cols, _ = infer_feature_columns(cfg, df)
    pre = fit_preprocessors(cfg, df, feature_cols)
    assert pre.model_features is not None
    assert pre.similarity_features is not None
