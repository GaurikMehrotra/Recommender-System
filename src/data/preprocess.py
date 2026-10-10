"""Load stable IDs and item metadata; retain sparse interaction storage."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

GENRES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def load_movielens(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return zero-based events, item records, and genres + fixed-scale year features.

    The full item catalog is treated as metadata known at evaluation time. No rating
    statistics or fitted scaler use validation/test events. This is a transductive
    catalog assumption, not a simulation of historical movie availability.
    """
    path = Path(path)
    if (path / "u.data").exists():
        ratings = pd.read_csv(
            path / "u.data", sep="\t", names=["user", "item", "rating", "timestamp"]
        )
        raw = pd.read_csv(path / "u.item", sep="|", encoding="latin-1", header=None)
        items = pd.DataFrame({"item": raw[0] - 1, "title": raw[1]})
        genres = raw.iloc[:, 5:24].to_numpy(dtype=np.float32)
    else:
        ratings = pd.read_csv(
            path / "ratings.dat",
            sep="::",
            engine="python",
            names=["user", "item", "rating", "timestamp"],
        )
        raw = pd.read_csv(
            path / "movies.dat",
            sep="::",
            engine="python",
            encoding="latin-1",
            names=["raw_id", "title", "genres"],
        )
        item_map = {int(v): i for i, v in enumerate(raw.raw_id)}
        ratings.item = ratings.item.map(item_map) + 1
        items = pd.DataFrame({"item": np.arange(len(raw)), "title": raw.title})
        genres = np.array(
            [[float(g in value.split("|")) for g in GENRES] for value in raw.genres],
            dtype=np.float32,
        )
    ratings.user = ratings.user - 1
    ratings.item = ratings.item - 1
    year = pd.to_numeric(items.title.str.extract(r"\((\d{4})\)")[0], errors="coerce")
    features = np.column_stack([genres, ((year.fillna(1990) - 1990) / 100).to_numpy()]).astype(
        np.float32
    )
    items["genres"] = ["|".join(g for g, x in zip(GENRES, row) if x) for row in genres]
    return ratings.astype({"user": "int64", "item": "int64"}), items, features


def interaction_matrix(
    events: pd.DataFrame, shape: tuple[int, int], binary: bool = True
) -> sparse.csr_matrix:
    """Build CSR without ever allocating a dense user-by-item array."""
    values = (
        np.ones(len(events), dtype=np.float32)
        if binary
        else events.rating.to_numpy(dtype=np.float32)
    )
    result = sparse.csr_matrix((values, (events.user, events.item)), shape=shape)
    if binary:
        result.data[:] = 1
    return result
