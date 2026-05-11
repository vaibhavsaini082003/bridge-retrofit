# Configuration

This project uses a single YAML file (see `configs/default.yaml`) to control:

- Dataset location (Excel/CSV)
- Target column names
- Which columns are excluded from features
- Preprocessing options (one-hot frequency threshold, optional SVD for similarity)
- Training options (model choice, split seed)

## Paths

- `project_root`: base folder for all relative paths.
  - Local: leave as `.`
  - Colab: set to your Drive mount, e.g. `/content/drive/MyDrive/retrofit`

## Columns

- `severity_target`: classification target for severity
- `retrofit_target`: retrofit label used by similarity-voting (and optional retrofit model)
- `exclude_features`: IDs/names/notes you don't want used as model features
- `case_display`: columns shown in the similar-case output table

## Similarity embedding

Similarity retrieval uses the same base encoding as modeling, plus an optional `TruncatedSVD` step to create a compact embedding.

- `preprocessing.svd_components: 0` disables SVD
- `preprocessing.svd_components: 50` is a reasonable default for one-hot + numeric features

## Large datasets

For ~100k rows, the most common bottleneck is categorical feature explosion.

- `preprocessing.onehot_min_frequency`: groups rare categories into an infrequent bucket.
- `preprocessing.onehot_max_categories`: optional hard cap per categorical column.

## Similarity backend

By default, similarity uses `sklearn` exact neighbors. For faster retrieval on large datasets, you can use FAISS:

- `similarity.backend: sklearn | faiss`
- `similarity.backend: faiss` requires `preprocessing.svd_components > 0` (dense embedding).

## GPU (optional)

This project is designed to run on free Colab CPU by default. If you have access to a GPU runtime (Colab GPU or your own server), you can optionally use GPU-accelerated training with XGBoost:

- Set `training.severity_model: xgb`
- Set `training.device: gpu`

Note: preprocessing (pandas/scikit-learn) remains CPU-bound; GPU mostly helps during model training.
