"""Baselines with configurable popularity temperature and seeded random scores."""

import numpy as np
from scipy import sparse


class Popularity:
    """Positive-event popularity; temperature is tuned using the same trial budget."""

    def __init__(self, temperature: float = 1.0, **kwargs: object) -> None:
        self.temperature = temperature

    def fit(self, matrix: sparse.csr_matrix) -> "Popularity":
        self.values = np.asarray(matrix.sum(axis=0)).ravel() ** self.temperature
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.values, (len(users), len(self.values)))


class Random:
    """Per-user deterministic random scores independent of evaluation batch size."""

    def __init__(self, seed: int = 42, **kwargs: object) -> None:
        self.seed = seed

    def fit(self, matrix: sparse.csr_matrix) -> "Random":
        self.n_items = matrix.shape[1]
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.random.default_rng(np.random.SeedSequence([self.seed, int(u)])).random(
                    self.n_items
                )
                for u in users
            ],
            dtype=np.float32,
        )
