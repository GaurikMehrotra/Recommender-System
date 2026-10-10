"""Export compact retrieval artifacts and serve full-catalog brute-force dot products."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.preprocessing import normalize

from src.explain import because_you_liked
from src.train import NEURAL, prepare


def export(checkpoint: str | Path, destination: str | Path) -> None:
    """Export learned MF/BPR/two-tower embeddings; NCF has no dot-product retrieval."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    name, config = state["model_name"], state["config"]
    if name not in ["mf", "bpr", "two_tower"]:
        raise ValueError("Export supports MF, BPR and two-tower, not nonlinear NCF")
    _, items, features, seen, positives, user_features = prepare(config)
    model = NEURAL[name](
        seen.shape[0],
        seen.shape[1],
        **config["params"],
        features=features,
        user_features=user_features,
    )
    model.load_state_dict(state["model"])
    model.eval()
    with torch.no_grad():
        if name == "two_tower":
            item_vectors = model.item_tower(model.features).numpy()
            user_vectors = model.user_tower(model.user_features).numpy()
            # Save learned user-tower weights for exact new-history inference.
            w1 = model.user_tower[0].weight.numpy()
            b1 = model.user_tower[0].bias.numpy()
            w2 = model.user_tower[3].weight.numpy()
            b2 = model.user_tower[3].bias.numpy()
            bias = np.zeros(seen.shape[1], dtype=np.float32)
        else:
            item_vectors = model.item_embedding.weight.numpy()
            user_vectors = model.user_embedding.weight.numpy()
            bias = model.item_bias.weight.squeeze(-1).numpy()
            w1, b1, w2, b2 = [np.empty(0, dtype=np.float32)] * 4
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "retrieval.npz",
        item_vectors=item_vectors,
        user_vectors=user_vectors,
        bias=bias,
        features=features,
        popularity=np.asarray(positives.sum(axis=0)).ravel(),
        indptr=seen.indptr,
        indices=seen.indices,
        positive_indptr=positives.indptr,
        positive_indices=positives.indices,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
    )
    (destination / "catalog.json").write_text(
        json.dumps(
            {
                "model": name,
                "titles": items.title.tolist(),
                "dataset_kind": "synthetic" if "synthetic" in config["data_path"] else "MovieLens",
            }
        )
    )


class Recommender:
    """Load only exported arrays; per-request allocation is O(items + embedding size)."""

    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        with np.load(directory / "retrieval.npz", allow_pickle=False) as arrays:
            self.arrays = {key: arrays[key] for key in arrays.files}
        metadata = json.loads((directory / "catalog.json").read_text())
        self.titles, self.model = metadata["titles"], metadata["model"]
        self.dataset_kind = metadata["dataset_kind"]

    def recommend(
        self, user: int | None = None, liked: list[int] | None = None, k: int = 10
    ) -> list[dict]:
        """Known IDs use learned user vectors; selected-movie users use content history.

        MF new-user fallback is explicitly content cosine, not a learned MF user.
        Two-tower new-user scoring uses its learned content-history user tower.
        """
        a = self.arrays
        history = list(dict.fromkeys(liked or []))
        explanation_history = history.copy()
        if k < 1 or k > 100:
            raise ValueError("k must be between 1 and 100")
        if any(i < 0 or i >= len(self.titles) for i in history):
            raise ValueError("Unknown item ID")
        if user is not None:
            if not 0 <= user < len(a["user_vectors"]):
                raise ValueError("Unknown user ID")
            history = a["indices"][a["indptr"][user] : a["indptr"][user + 1]].tolist()
            explanation_history = a["positive_indices"][
                a["positive_indptr"][user] : a["positive_indptr"][user + 1]
            ].tolist()
            scores = a["item_vectors"] @ a["user_vectors"][user] + a["bias"]
        elif history and self.model == "two_tower":
            content = a["features"][history].mean(axis=0)
            vector = np.maximum(content @ a["w1"].T + a["b1"], 0) @ a["w2"].T + a["b2"]
            scores = a["item_vectors"] @ vector
        elif history:
            content = normalize(a["features"][history]).mean(axis=0)
            scores = normalize(a["features"]) @ content
        else:
            scores = a["popularity"].copy()
        scores = scores.copy()
        scores[history] = -np.inf
        ranked = np.argsort(-scores, kind="stable")[:k]
        return [
            {
                "item": int(i),
                "title": self.titles[i],
                "score": float(scores[i]),
                "explanation": because_you_liked(
                    int(i), explanation_history, a["features"], self.titles
                ),
            }
            for i in ranked
            if np.isfinite(scores[i])
        ]

    def benchmark(self, requests: int = 100) -> dict:
        """Warm latency distribution and process RSS (not isolated artifact memory)."""
        self.recommend(user=0)
        times = []
        for i in range(requests):
            start = time.perf_counter()
            self.recommend(user=i % len(self.arrays["user_vectors"]))
            times.append((time.perf_counter() - start) * 1000)
        return {
            "requests": requests,
            "p50_ms": float(np.median(times)),
            "p95_ms": float(np.quantile(times, 0.95)),
            "process_rss_mb": psutil.Process().memory_info().rss / 1024**2,
            "array_mb": sum(a.nbytes for a in self.arrays.values()) / 1024**2,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--artifacts", default="artifacts/demo")
    parser.add_argument("--user", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.checkpoint:
        export(args.checkpoint, args.artifacts)
    model = Recommender(args.artifacts)
    print(json.dumps(model.benchmark() if args.benchmark else model.recommend(args.user), indent=2))


if __name__ == "__main__":
    main()
