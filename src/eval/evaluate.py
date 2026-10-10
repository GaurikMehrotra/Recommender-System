"""Full-catalog ranking in bounded user batches, with deterministic tie breaking."""

import numpy as np
import pandas as pd
from scipy import sparse

from src.eval.metrics import ranking_metrics
from src.models.base import ExplicitModel, RankingModel


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Stable descending ranking; exact ties favor lower internal item IDs."""
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def evaluate(
    model: RankingModel,
    truth: pd.DataFrame,
    seen: sparse.csr_matrix,
    ks: tuple[int, ...] | list[int] = (10, 20),
    batch_size: int = 64,
    extra_seen: sparse.csr_matrix | None = None,
    novelty_penalty: float = 0.0,
    popularity: np.ndarray | None = None,
) -> tuple[dict, pd.DataFrame, dict]:
    """Score ALL catalog items, excluding all prior observed ratings.

    A batch-by-catalog score array is permitted; a full users-by-items array is not.
    For final test also exclude validation history (extra_seen), without fitting on it.
    """
    relevant = {int(u): set(g.item.astype(int)) for u, g in truth.groupby("user")}
    users = np.array(sorted(relevant), dtype=np.int64)
    records, rankings = [], {}
    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        scores = np.asarray(model.score(batch), dtype=np.float32).copy()
        if scores.shape != (len(batch), seen.shape[1]) or not np.isfinite(scores).all():
            # Restricted cold-item scorers deliberately return -inf outside their universe.
            if (
                scores.shape != (len(batch), seen.shape[1])
                or np.isnan(scores).any()
                or np.isposinf(scores).any()
            ):
                raise ValueError("Invalid full-catalog score shape or nonfinite model scores")
        if novelty_penalty:
            if popularity is None:
                raise ValueError("Reranking requires train-only popularity")
            scores -= novelty_penalty * np.log1p(popularity)[None, :]
        for row, user in enumerate(batch):
            scores[row, seen.getrow(int(user)).indices] = -np.inf
            if extra_seen is not None:
                scores[row, extra_seen.getrow(int(user)).indices] = -np.inf
        ordered = topk(scores, max(ks))
        for row, user in enumerate(batch):
            ranking = ordered[row][np.isfinite(scores[row, ordered[row]])]
            rankings[int(user)] = ranking.tolist()
            record = {"user": int(user)}
            for k in ks:
                record.update(ranking_metrics(ranking, relevant[int(user)], k))
            records.append(record)
    per_user = pd.DataFrame(records)
    if per_user.empty:
        raise ValueError("No eligible evaluation users")
    return per_user.drop(columns="user").mean().to_dict(), per_user, rankings


def explicit_metrics(model: ExplicitModel, events: pd.DataFrame) -> dict[str, float]:
    """Evaluate observed held-out positive ratings; report this selection bias."""
    if not hasattr(model, "predict_pairs"):
        return {}
    prediction = model.predict_pairs(events.user.to_numpy(), events.item.to_numpy())
    error = prediction - events.rating.to_numpy()
    return {"RMSE": float(np.sqrt(np.mean(error**2))), "MAE": float(np.mean(np.abs(error)))}
