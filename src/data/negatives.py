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
        """Rejection sampling with bounded retries and exact fallback."""
        output = np.empty(len(users), dtype=np.int64)
        for index, user in enumerate(users):
            seen = self.positives.getrow(int(user)).indices
            candidate = int(self.rng.choice(self.candidates))
            attempts = 0
            while candidate in seen:
                attempts += 1
                if attempts >= 32:
                    available = np.setdiff1d(self.candidates, seen)
                    if not len(available):
                        raise ValueError("User has no unobserved items")
                    candidate = int(self.rng.choice(available))
                    break
                candidate = int(self.rng.choice(self.candidates))
            output[index] = candidate
        return output
