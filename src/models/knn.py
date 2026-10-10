"""Sparse cosine kNN with blockwise top-neighbor pruning."""

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize


def sparse_similarity(
    matrix: sparse.csr_matrix, k: int, shrinkage: float = 10.0, block_size: int = 128
) -> sparse.csr_matrix:
    """Never materialize a dense similarity matrix; retain at most k edges per row."""
    normalized = normalize(matrix, axis=1).tocsr()
    rows, cols, values = [], [], []
    for start in range(0, matrix.shape[0], block_size):
        similarities = (normalized[start : start + block_size] @ normalized.T).tocsr()
        counts = (matrix[start : start + block_size] @ matrix.T).tocsr()
        counts.data = counts.data / (counts.data + shrinkage)
        similarities = similarities.multiply(counts).tocsr()
        for local in range(similarities.shape[0]):
            row = similarities.getrow(local)
            mask = row.indices != start + local
            ids, scores = row.indices[mask], row.data[mask]
            keep = np.argsort(-scores, kind="stable")[:k]
            rows.extend([start + local] * len(keep))
            cols.extend(ids[keep])
            values.extend(scores[keep])
    return sparse.csr_matrix(
        (values, (rows, cols)), shape=(matrix.shape[0], matrix.shape[0]), dtype=np.float32
    )


class KNN:
    """Weighted implicit neighborhood scores for user or item cosine neighbors."""

    def __init__(
        self, mode: str = "item", k: int = 40, shrinkage: float = 10, **kwargs: object
    ) -> None:
        self.mode, self.k, self.shrinkage = mode, k, shrinkage

    def fit(self, matrix: sparse.csr_matrix) -> "KNN":
        self.matrix = matrix.tocsr()
        target = self.matrix.T.tocsr() if self.mode == "item" else self.matrix
        self.similarity = sparse_similarity(target, self.k, self.shrinkage)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        result = (
            self.matrix[users] @ self.similarity
            if self.mode == "item"
            else self.similarity[users] @ self.matrix
        )
        return result.toarray()
