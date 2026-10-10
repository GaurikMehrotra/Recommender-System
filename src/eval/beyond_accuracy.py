"""Train-only popularity and content-based diversity, with explicit definitions."""

import numpy as np
from sklearn.preprocessing import normalize


def beyond_accuracy(
    rankings: dict,
    popularity: np.ndarray,
    features: np.ndarray,
    k: int = 20,
    head_fraction: float = 0.2,
) -> dict[str, float]:
    """Coverage/catalog size, self-information novelty, genre/year cosine ILD, head share.

    Add-one novelty smoothing includes zero-interaction catalog items. Diversity is
    based on provided content features and is not a claim about perceived diversity.
    """
    lists = [np.asarray(row[:k], dtype=int) for row in rankings.values() if len(row)]
    if not lists:
        raise ValueError("No recommendations")
    ids = np.concatenate(lists)
    probability = (popularity + 1) / (popularity.sum() + len(popularity))
    head = np.argsort(-popularity, kind="stable")[
        : max(1, int(np.ceil(len(popularity) * head_fraction)))
    ]
    normalized = normalize(features)
    diversity = []
    for row in lists:
        if len(row) > 1:
            distance = 1 - normalized[row] @ normalized[row].T
            diversity.append(float(distance[np.triu_indices(len(row), 1)].mean()))
    return {
        "coverage": float(len(np.unique(ids)) / len(popularity)),
        "novelty_bits": float((-np.log2(probability[ids])).mean()),
        "intra_list_diversity": float(np.mean(diversity)) if diversity else 0.0,
        "head_share": float(np.isin(ids, head).mean()),
    }
