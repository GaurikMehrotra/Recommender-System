"""Validation-only cold-start simulations with matched candidate universes."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import interaction_matrix
from src.eval.evaluate import evaluate
from src.models.mf import fold_in_user
from src.train import fit, prepare
from src.utils import load_config


def run(config: dict, output: Path) -> None:
    """Disjoint held users/items; adapt user history without optimizing their ID vectors.

    New users: remove all their events when fitting; reveal earliest 1-5 TRAIN
    positives only at inference. Compare content tower vs MF ridge fold-in
    (MF fits a new user factor only from the revealed ratings, never validation labels).
    New items: remove their events when fitting; score the FULL catalog for
    both models, conditioning diagnostic truth on held-item positives. MF item vectors are untrained; two-tower uses item metadata.
    Neither experiment uses final-test labels.
    """
    prepared = prepare(config)
    split, items, features, seen, positives, _ = prepared
    rng = np.random.default_rng(config["seed"])
    eligible_counts = (
        split.train[split.train.rating >= config["positive_threshold"]].groupby("user").size()
    )
    candidates = np.intersect1d(
        split.validation.user.unique(), eligible_counts[eligible_counts >= 5].index
    )
    if len(candidates) < 2:
        raise ValueError("Cold-user experiment needs at least two users with five train positives")
    held_users = rng.choice(
        candidates, max(1, int(len(candidates) * config.get("cold_fraction", 0.1))), replace=False
    )
    held_items = rng.choice(
        len(items), max(1, int(len(items) * config.get("cold_fraction", 0.1))), replace=False
    )
    rows = []
    for mode in ["users", "items"]:
        local = copy.deepcopy(split)
        local.train = local.train[
            ~(
                local.train.user.isin(held_users)
                if mode == "users"
                else local.train.item.isin(held_items)
            )
        ]
        # The diagnostic cohort must not select its own early-stop epoch.
        local.validation = local.validation[
            ~(
                local.validation.user.isin(held_users)
                if mode == "users"
                else local.validation.item.isin(held_items)
            )
        ]
        local_config = copy.deepcopy(config)
        if mode == "items":
            # No training feedback of either sign for cold items.
            local_config["excluded_training_items"] = held_items.tolist()
        matrix = interaction_matrix(local.train, seen.shape)
        pos = interaction_matrix(
            local.train[local.train.rating >= config["positive_threshold"]], seen.shape
        )
        count = np.asarray(pos.sum(axis=1)).ravel()
        uf = np.asarray(pos @ features) / np.maximum(count[:, None], 1)
        modified = (local, items, features, matrix, pos, uf)
        for name in ["mf", "two_tower"]:
            scorer, _ = fit(local_config, name, output / mode / name, prepared=modified)
            if mode == "items":
                truth = split.validation[split.validation.item.isin(held_items)]
                if truth.empty:
                    rows.append({"mode": mode, "model": name, "status": "no eligible positives"})
                    continue

                # Fit sees no cold-item feedback, but the original historical mask
                # still prevents recommending movies already observed by that user.
                metrics, _, _ = evaluate(
                    scorer, truth, seen, config["ks"], config["eval_batch_size"]
                )
                rows.append(
                    {"mode": mode, "model": name, "users": int(truth.user.nunique()), **metrics}
                )
            else:
                truth = split.validation[split.validation.user.isin(held_users)]
                for history_size in range(1, 6):
                    histories = (
                        split.train[
                            (split.train.user.isin(held_users))
                            & (split.train.rating >= config["positive_threshold"])
                        ]
                        .sort_values("timestamp")
                        .groupby("user")
                        .head(history_size)
                    )
                    history_matrix = interaction_matrix(histories, seen.shape)
                    history_ratings = interaction_matrix(histories, seen.shape, binary=False)
                    import torch

                    model = scorer.model

                    class Adapted:
                        def score(self, users: np.ndarray) -> np.ndarray:
                            batch_history = history_matrix[users]
                            counts = np.asarray(batch_history.sum(axis=1)).ravel()
                            with torch.no_grad():
                                if name == "two_tower":
                                    content = np.asarray(batch_history @ features) / np.maximum(
                                        counts[:, None], 1
                                    )
                                    vector = model.user_tower(
                                        torch.as_tensor(
                                            content, dtype=torch.float32, device=scorer.device
                                        )
                                    )
                                    item_vec = model.item_tower(model.features)
                                    return (vector @ item_vec.T).cpu().numpy()
                                item_vec = model.item_embedding.weight.detach().cpu().numpy()
                                bias = model.item_bias.weight.detach().cpu().numpy().ravel()
                                mean = float(model.global_mean.cpu())
                                factors = []
                                user_bias = []
                                for user in users:
                                    row = history_ratings.getrow(int(user))
                                    vector, offset = fold_in_user(
                                        item_vec[row.indices],
                                        row.data,
                                        bias[row.indices],
                                        mean,
                                        config.get("cold_regularization", config["weight_decay"]),
                                    )
                                    factors.append(vector)
                                    user_bias.append(offset)
                                return (
                                    np.asarray(factors) @ item_vec.T
                                    + bias[None, :]
                                    + mean
                                    + np.asarray(user_bias)[:, None]
                                )

                    metrics, _, _ = evaluate(
                        Adapted(), truth, history_matrix, config["ks"], config["eval_batch_size"]
                    )
                    rows.append(
                        {
                            "mode": mode,
                            "model": name,
                            "history_size": history_size,
                            "users": int(truth.user.nunique()),
                            **metrics,
                        }
                    )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "cold_start.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "candidate_universe": "full_catalog",
                "held_users": held_users.tolist(),
                "held_items": held_items.tolist(),
                "config": config,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cold_start.yaml")
    parser.add_argument("--output", default="runs/cold_start")
    args = parser.parse_args()
    run(load_config(args.config), Path(args.output))


if __name__ == "__main__":
    main()
