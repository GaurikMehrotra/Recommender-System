"""GMF, MLP, and NeuMF implemented directly in PyTorch."""

import torch
from torch import nn


class NCF(nn.Module):
    """Pointwise BCE with uniformly sampled unobserved negatives + L2.

    NeuMF uses separate GMF and MLP embeddings; no pretrained weights are assumed.
    """

    explicit = False

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        variant: str = "neumf",
        dropout: float = 0.1,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.mlp_user = nn.Embedding(n_users, embedding_dim)
        self.mlp_item = nn.Embedding(n_items, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )
        self.output = nn.Linear(embedding_dim * (2 if variant == "neumf" else 1), 1)
        for layer in [self.user_embedding, self.item_embedding, self.mlp_user, self.mlp_item]:
            nn.init.normal_(layer.weight, std=0.05)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        if self.variant == "gmf":
            features = self.user_embedding(users) * self.item_embedding(items)
        else:
            mlp = self.mlp(torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=-1))
            if self.variant == "mlp":
                features = mlp
            else:
                gmf = self.user_embedding(users) * self.item_embedding(items)
                features = torch.cat([gmf, mlp], dim=-1)
        return self.output(features).squeeze(-1)
