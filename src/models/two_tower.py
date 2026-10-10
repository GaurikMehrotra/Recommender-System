"""Content-aware two-tower retrieval with history-pooled users and content-only items."""

import numpy as np
import torch
from torch import nn


class TwoTower(nn.Module):
    """BCE dot-product retrieval; content-only item vectors generalize to unseen items.

    User tower consumes mean training-history genre/year features, not a user ID
    embedding. Thus users with 1-5 selected movies can use the same learned tower.
    """

    explicit = False

    def __init__(
        self,
        n_users: int,
        n_items: int,
        features: np.ndarray,
        user_features: np.ndarray,
        embedding_dim: int = 32,
        dropout: float = 0.1,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.register_buffer("features", torch.as_tensor(features, dtype=torch.float32))
        self.register_buffer("user_features", torch.as_tensor(user_features, dtype=torch.float32))
        dim = self.features.shape[1]
        self.user_tower = nn.Sequential(
            nn.Linear(dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (
            self.user_tower(self.user_features[users]) * self.item_tower(self.features[items])
        ).sum(-1)
