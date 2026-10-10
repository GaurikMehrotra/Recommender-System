"""Transparent content similarity explanations; not causal model attribution."""

import numpy as np
from sklearn.preprocessing import normalize


def because_you_liked(
    item: int, history: list[int], features: np.ndarray, titles: list[str]
) -> str:
    """Name the most content-similar liked title, disclosing explanation provenance."""
    if not history:
        return "Popular among training users; no personal history provided."
    normalized = normalize(features)
    similarity = normalized[history] @ normalized[item]
    best = history[int(np.argmax(similarity))]
    return f"Because you liked {titles[best]} (genre/year similarity; not a causal attribution)."
