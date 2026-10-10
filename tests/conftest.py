"""Tiny deterministic synthetic MovieLens-format fixture, never real user data."""

import pandas as pd
import pytest

from src.utils import load_config


@pytest.fixture
def tiny(tmp_path):
    path = tmp_path / "ml-100k"
    path.mkdir()
    rows = []
    for user in range(1, 9):
        for t in range(8):
            rows.append([user, (user * 3 + t) % 32 + 1, 4 + t % 2, 100 + t])
    pd.DataFrame(rows).to_csv(path / "u.data", sep="\t", header=False, index=False)
    movies = []
    for item in range(1, 33):
        genres = [int(g == item % 19) for g in range(19)]
        movies.append([item, f"Synthetic {item} (1995)", "01-Jan-1995", "", "", *genres])
    pd.DataFrame(movies).to_csv(path / "u.item", sep="|", header=False, index=False)
    config = load_config("configs/debug_cpu.yaml")
    config.update(data_path=str(path), epochs=2, patience=2, batch_size=32, threads=1)
    return config
