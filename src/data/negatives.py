"""Uniform train-unobserved sampling with optional cold-item exclusions."""

import numpy as np
from scipy import sparse


class NegativeSampler:
    """Held-out positive labels remain unknown; simulated cold items may be excluded."""

    def __init__(
        self, positives: sparse.csr_matrix, seed: int = 42, excluded_items: list[int] | None = None
    ) -> None:
        self.positives = positives.tocsr()
        self.rng = np.random.default_rng(seed)
        self.candidates = np.setdiff1d(np.arange(positives.shape[1]), excluded_items or [])
        if not len(self.candidates):
            raise ValueError("No eligible negative items")

    def sample(self, users: np.ndarray) -> np.ndarray:
        """Vectorized CSR rejection sampling, with a bounded saturated-user fallback."""
        users = np.asarray(users, dtype=np.int64)
        output = self.rng.choice(self.candidates, size=len(users)).astype(np.int64)
        for _ in range(32):
            invalid = np.asarray(self.positives[users, output]).ravel() > 0
            if not invalid.any():
                return output
            output[invalid] = self.rng.choice(self.candidates, size=int(invalid.sum()))
        invalid = np.asarray(self.positives[users, output]).ravel() > 0
        for index in np.flatnonzero(invalid):
            user = users[index]
            seen = self.positives.indices[
                self.positives.indptr[user] : self.positives.indptr[user + 1]
            ]
            available = np.setdiff1d(self.candidates, seen)
            if not len(available):
                raise ValueError("User has no unobserved items")
            output[index] = self.rng.choice(available)
        return output
