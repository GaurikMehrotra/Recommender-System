"""Ranking metrics with binary relevance, macro-averaged over eligible users."""

import numpy as np


def ranking_metrics(ranking: np.ndarray, relevant: set[int], k: int) -> dict[str, float]:
    """AP divides by min(K, number relevant); empty truth has defined zero metrics."""
    hits = np.array([int(i) in relevant for i in ranking[:k]], dtype=float)
    count = len(relevant)
    ideal = np.sum(1 / np.log2(np.arange(min(k, count)) + 2))
    ap = np.sum(np.cumsum(hits) / (np.arange(len(hits)) + 1) * hits)
    return {
        f"Recall@{k}": float(hits.sum() / count) if count else 0.0,
        f"NDCG@{k}": float(np.sum(hits / np.log2(np.arange(len(hits)) + 2)) / ideal)
        if ideal
        else 0.0,
        f"MAP@{k}": float(ap / min(k, count)) if count else 0.0,
        f"HitRate@{k}": float(hits.any()),
    }
