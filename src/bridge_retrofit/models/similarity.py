"""Similarity retrieval + retrofit recommendation via voting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from bridge_retrofit.config import ProjectConfig
from bridge_retrofit.preprocessing.pipeline import Preprocessors


@dataclass(frozen=True)
class NeighborResult:
    indices: np.ndarray
    distances: np.ndarray


class SimilarityIndex:
    def __init__(self, nn: NearestNeighbors, matrix):
        self._nn = nn
        self._matrix = matrix

    def query(self, vector, k: int) -> NeighborResult:
        # Ensure 2D query shape for both sparse and dense vectors.
        if hasattr(vector, "ndim") and getattr(vector, "ndim") == 1:
            vector = vector.reshape(1, -1)
        distances, indices = self._nn.kneighbors(vector, n_neighbors=k)
        return NeighborResult(indices=indices[0], distances=distances[0])


class FaissSimilarityIndex:
    """Optional FAISS-backed index.

    Stores the FAISS index in serialized form so it remains joblib-pickleable.
    """

    def __init__(self, serialized_index: np.ndarray, dim: int):
        self._serialized_index = serialized_index
        self._dim = int(dim)

    def _load(self):
        try:
            import faiss  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("faiss is not installed; install faiss-cpu (Colab) or switch similarity.backend to 'sklearn'") from e
        return faiss.deserialize_index(self._serialized_index)

    def query(self, vector, k: int) -> NeighborResult:
        import numpy as _np

        x = vector
        if hasattr(x, "toarray"):
            x = x.toarray()
        x = _np.asarray(x, dtype=_np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self._dim:
            raise ValueError(f"Query dim mismatch: expected {self._dim}, got {x.shape[1]}")

        index = self._load()
        distances, indices = index.search(x, int(k))
        return NeighborResult(indices=indices[0], distances=distances[0])


def fit_similarity_index(
    cfg: ProjectConfig,
    df: pd.DataFrame,
    preprocessors: Preprocessors,
    feature_cols: list[str],
) -> tuple[SimilarityIndex, pd.DataFrame]:
    backend = (cfg.similarity.backend or "sklearn").lower()
    X = preprocessors.similarity_features.transform(df[feature_cols])

    if backend == "sklearn":
        nn = NearestNeighbors(n_neighbors=cfg.similarity.n_neighbors, metric="euclidean", n_jobs=-1)
        nn.fit(X)
        index: SimilarityIndex | FaissSimilarityIndex = SimilarityIndex(nn, X)
    elif backend == "faiss":
        if not (cfg.preprocessing.svd_components and cfg.preprocessing.svd_components > 0):
            raise ValueError("similarity.backend='faiss' requires preprocessing.svd_components > 0 (dense embedding)")
        try:
            import faiss  # type: ignore
        except Exception as e:
            raise RuntimeError("faiss is not installed; install faiss-cpu (recommended in Colab)") from e

        x_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        x_dense = np.asarray(x_dense, dtype=np.float32)
        dim = int(x_dense.shape[1])
        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(x_dense)
        serialized = faiss.serialize_index(faiss_index)
        index = FaissSimilarityIndex(serialized, dim=dim)
    else:
        raise ValueError(f"Unknown similarity backend: {cfg.similarity.backend}")

    display_cols = [c for c in cfg.columns.case_display if c in df.columns]
    # Always include retrofit column (needed for voting), if present.
    if cfg.columns.retrofit_target in df.columns and cfg.columns.retrofit_target not in display_cols:
        display_cols.append(cfg.columns.retrofit_target)
    case_table = df[display_cols].copy() if display_cols else df[[cfg.columns.retrofit_target]].copy()

    return index, case_table


def recommend_retrofit_from_neighbors(neighbor_rows: pd.DataFrame, retrofit_col: str) -> str | None:
    if retrofit_col not in neighbor_rows.columns:
        return None
    counts = neighbor_rows[retrofit_col].astype(str).value_counts(dropna=True)
    if counts.empty:
        return None
    return str(counts.index[0])
