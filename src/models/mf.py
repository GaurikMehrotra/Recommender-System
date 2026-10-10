"""PyTorch explicit matrix factorization with user/item biases."""

import torch
from torch import nn


class MF(nn.Module):
    """Prediction mu + b_u + b_i + p_u dot q_i; optimized with MSE + L2."""

    explicit = True

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        global_mean: float = 3.5,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.register_buffer("global_mean", torch.tensor(global_mean))
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (
            (self.user_embedding(users) * self.item_embedding(items)).sum(-1)
            + self.user_bias(users).squeeze(-1)
            + self.item_bias(items).squeeze(-1)
            + self.global_mean
        )
