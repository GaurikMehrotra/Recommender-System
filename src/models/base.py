"""Structural interfaces shared by classical and neural ranking models."""

from typing import Protocol

import numpy as np


class RankingModel(Protocol):
    """Any fitted model that scores a user batch against the complete catalog."""

    def score(self, users: np.ndarray) -> np.ndarray:
        """Return batch-users by catalog scores."""
        ...


class ExplicitModel(RankingModel, Protocol):
    """A scorer supporting observed pair predictions."""

    def predict_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Return one prediction per observed pair."""
        ...
