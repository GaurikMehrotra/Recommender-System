"""Compact tensor datasets for explicit and implicit feedback."""

import pandas as pd
import torch
from torch.utils.data import TensorDataset


def rating_dataset(events: pd.DataFrame) -> TensorDataset:
    """Store only observed rows; no dense user-item matrix."""
    return TensorDataset(
        torch.tensor(events.user.to_numpy(), dtype=torch.long),
        torch.tensor(events.item.to_numpy(), dtype=torch.long),
        torch.tensor(events.rating.to_numpy(), dtype=torch.float32),
    )
