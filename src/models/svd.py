"""TruncatedSVD on a sparse implicit interaction matrix."""

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


class SVD:
    """Uncentered sparse low-rank reconstruction; not explicit-rating SVD++."""

    def __init__(self, embedding_dim: int = 32, seed: int = 42, **kwargs: object) -> None:
        self.dim, self.seed = embedding_dim, seed

    def fit(self, matrix: sparse.csr_matrix) -> "SVD":
        model = TruncatedSVD(
            n_components=min(self.dim, min(matrix.shape) - 1), random_state=self.seed
        )
        self.users = model.fit_transform(matrix)
        self.items = model.components_.T
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.users[users] @ self.items.T
