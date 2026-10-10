"""Unified sparse training, validation early stopping, and resumable checkpoints."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.tensorboard import SummaryWriter

from src.data.negatives import NegativeSampler
from src.data.preprocess import interaction_matrix, load_movielens
from src.data.splits import Split, assert_no_leakage, leave_last_out
from src.eval.evaluate import evaluate, explicit_metrics
from src.models.base import RankingModel
from src.models.baselines import Popularity, Random
from src.models.bpr import BPR
from src.models.knn import KNN
from src.models.mf import MF
from src.models.ncf import NCF
from src.models.svd import SVD
from src.models.two_tower import TwoTower
from src.utils import data_fingerprint, device_for, load_config, seed_everything

NEURAL = {"mf": MF, "bpr": BPR, "gmf": NCF, "mlp": NCF, "neumf": NCF, "two_tower": TwoTower}
MODELS = ["random", "popularity", "user_knn", "item_knn", "svd", *NEURAL]


class TorchScorer:
    """Bounded item blocks for NCF; matrix multiplication for retrieval models."""

    def __init__(
        self, model: torch.nn.Module, n_items: int, device: torch.device, item_batch_size: int = 256
    ) -> None:
        self.model, self.n_items, self.device = model, n_items, device
        self.item_batch_size = item_batch_size

    @torch.no_grad()
    def score(self, users: np.ndarray) -> np.ndarray:
        self.model.eval()
        u = torch.as_tensor(users, dtype=torch.long, device=self.device)
        if isinstance(self.model, NCF) and self.model.variant == "gmf":
            # Linear head on elementwise products is an exact weighted dot product.
            weighted_users = self.model.user_embedding(u) * self.model.output.weight[0]
            scores = weighted_users @ self.model.item_embedding.weight.T + self.model.output.bias[0]
            return scores.cpu().numpy()
        if isinstance(self.model, TwoTower):
            user_vec = self.model.user_tower(self.model.user_features[u])
            item_vec = self.model.item_tower(self.model.features)
            return (user_vec @ item_vec.T).cpu().numpy()
        if isinstance(self.model, MF):
            scores = self.model.user_embedding(u) @ self.model.item_embedding.weight.T
            scores += (
                self.model.user_bias(u) + self.model.item_bias.weight.T + self.model.global_mean
            )
            return scores.cpu().numpy()
        result = np.empty((len(users), self.n_items), dtype=np.float32)
        for start in range(0, self.n_items, self.item_batch_size):
            ids = torch.arange(
                start, min(start + self.item_batch_size, self.n_items), device=self.device
            )
            uu = u[:, None].expand(-1, len(ids)).reshape(-1)
            ii = ids[None, :].expand(len(u), -1).reshape(-1)
            result[:, start : start + len(ids)] = (
                self.model(uu, ii).reshape(len(u), -1).cpu().numpy()
            )
        return result

    @torch.no_grad()
    def predict_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        self.model.eval()
        return (
            self.model(
                torch.as_tensor(users, device=self.device),
                torch.as_tensor(items, device=self.device),
            )
            .cpu()
            .numpy()
        )


def prepare(
    config: dict,
) -> tuple[Split, pd.DataFrame, np.ndarray, sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """Load and split once, with train-only sparse statistics and history features."""
    events, items, features = load_movielens(config["data_path"])
    split = leave_last_out(events, config["positive_threshold"])
    assert_no_leakage(split)
    shape = (int(events.user.max()) + 1, len(items))
    seen = interaction_matrix(split.train, shape)
    positives = interaction_matrix(
        split.train[split.train.rating >= config["positive_threshold"]], shape
    )
    counts = np.asarray(positives.sum(axis=1)).ravel()
    user_features = np.asarray(positives @ features) / np.maximum(counts[:, None], 1)
    return split, items, features, seen, positives, user_features


def fit(
    config: dict,
    model_name: str,
    output: str | Path,
    resume: bool = False,
    prepared: tuple | None = None,
) -> tuple[RankingModel, dict[str, float]]:
    """Fit using validation only; checkpoint includes optimizer and RNG state.

    Explicit MF uses all pre-cutoff ratings; other models use positive ratings.
    Negative sampling excludes every TRAIN observed rating, including dislikes.
    """
    seed_everything(config["seed"], config["threads"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    split, items, features, seen, positives, user_features = prepared or prepare(config)
    import platform
    from importlib.metadata import version

    import yaml

    from src.utils import atomic_json

    (output / "config.yaml").write_text(yaml.safe_dump(config))
    atomic_json(
        output / "provenance.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in ["torch", "numpy", "pandas", "scipy", "scikit-learn"]
            },
            "shape": list(seen.shape),
            "data_fingerprint": data_fingerprint(config["data_path"]),
            "train_ratings": len(split.train),
            "train_positives": positives.nnz,
            "validation_positives": len(split.validation),
            "test_positives": len(split.test),
            "excluded_users": split.excluded_users,
            "seed": config["seed"],
            "model": model_name,
            "device": str(device_for(config["device"])),
        },
    )
    fingerprint = data_fingerprint(config["data_path"])
    params = config.get("params", {})
    if model_name not in NEURAL:
        cls = {
            "random": Random,
            "popularity": Popularity,
            "svd": SVD,
            "user_knn": KNN,
            "item_knn": KNN,
        }[model_name]
        kwargs = dict(params, seed=config["seed"])
        if model_name.endswith("knn"):
            kwargs["mode"] = model_name.split("_")[0]
        scorer = cls(**kwargs).fit(positives)
        metrics, _, _ = evaluate(
            scorer, split.validation, seen, config["ks"], config["eval_batch_size"]
        )
        return scorer, metrics
    device = device_for(config["device"])
    variant = {"variant": model_name} if model_name in ["gmf", "mlp", "neumf"] else {}
    model = NEURAL[model_name](
        seen.shape[0],
        seen.shape[1],
        **params,
        **variant,
        global_mean=float(split.train.rating.mean()),
        features=features,
        user_features=user_features,
    ).to(device)
    scorer = TorchScorer(model, seen.shape[1], device, config["item_batch_size"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    events = (
        split.train
        if model_name == "mf"
        else split.train[split.train.rating >= config["positive_threshold"]]
    )
    users, items_arr = events.user.to_numpy(), events.item.to_numpy()
    ratings = events.rating.to_numpy(dtype=np.float32)
    sampler = NegativeSampler(seen, config["seed"], config.get("excluded_training_items"))
    rng = np.random.default_rng(config["seed"])
    checkpoint = output / "last.pt"
    best_value, best_state, stale, first = -np.inf, None, 0, 0
    best_metrics = {}
    if resume:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if (
            state["config"] != config
            or state["model_name"] != model_name
            or state.get("data_fingerprint") != fingerprint
        ):
            raise ValueError("Resume requires identical config and model, including epoch budget")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best_value, best_state, best_metrics = (
            state["best_value"],
            state["best_state"],
            state["best_metrics"],
        )
        stale, first = state["stale"], state["epoch"] + 1
        rng.bit_generator.state = state["rng"]
        sampler.rng.bit_generator.state = state["sampler_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    writer = SummaryWriter(str(output / "tensorboard"))
    try:
        for epoch in range(first, config["epochs"]):
            if stale >= config["patience"]:
                break
            model.train()
            order = rng.permutation(len(events))
            losses = []
            for start in range(0, len(order), config["batch_size"]):
                idx = order[start : start + config["batch_size"]]
                uu = torch.as_tensor(users[idx], device=device)
                ii = torch.as_tensor(items_arr[idx], device=device)
                score = model(uu, ii)
                if model_name == "mf":
                    loss = torch.nn.functional.mse_loss(
                        score, torch.as_tensor(ratings[idx], device=device)
                    )
                else:
                    negative = torch.as_tensor(sampler.sample(users[idx]), device=device)
                    neg_score = model(uu, negative)
                    if model_name == "bpr":
                        loss = -torch.nn.functional.logsigmoid(score - neg_score).mean()
                    else:
                        # -log sigmoid(s+) - log(1-sigmoid(s-)); Adam weight decay adds L2.
                        loss = (
                            torch.nn.functional.softplus(-score).mean()
                            + torch.nn.functional.softplus(neg_score).mean()
                        ) / 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            metrics, _, _ = evaluate(
                scorer, split.validation, seen, config["ks"], config["eval_batch_size"]
            )
            value = metrics["NDCG@20"]
            writer.add_scalar("train/loss", float(np.mean(losses)), epoch)
            for key, number in metrics.items():
                writer.add_scalar(f"validation/{key}", number, epoch)
            if value > best_value:
                best_value, best_metrics, stale = value, metrics, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_value": best_value,
                    "best_state": best_state,
                    "best_metrics": best_metrics,
                    "stale": stale,
                    "rng": rng.bit_generator.state,
                    "sampler_rng": sampler.rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "data_fingerprint": fingerprint,
                    "config": config,
                    "model_name": model_name,
                },
                checkpoint.with_suffix(".pt.tmp"),
            )
            checkpoint.with_suffix(".pt.tmp").replace(checkpoint)
            print(
                f"{model_name} epoch={epoch + 1} loss={np.mean(losses):.4f} val_ndcg20={value:.4f}",
                flush=True,
            )
    finally:
        writer.close()
    model.load_state_dict(best_state)
    torch.save(
        {"model": best_state, "config": config, "model_name": model_name}, output / "best.pt"
    )
    return scorer, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/debug_cpu.yaml")
    parser.add_argument("--model", choices=MODELS, default="mf")
    parser.add_argument("--output", default="runs/debug")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Consume one-time test evaluation; never use for tuning",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    prepared = prepare(config)
    scorer, metrics = fit(config, args.model, args.output, args.resume, prepared)
    result = {"config": config, "model": args.model, "validation": metrics}
    if args.final_test:
        output = Path(args.output)
        marker = output / "TEST_CONSUMED"
        with marker.open("x") as handle:
            handle.write("Final test consumed. Do not retune after inspecting results.\n")
        split, _, _, seen, _, _ = prepared
        extra = interaction_matrix(split.validation, seen.shape)
        result["test"], per_user, rankings = evaluate(
            scorer, split.test, seen, config["ks"], config["eval_batch_size"], extra
        )
        if args.model == "mf":
            result["test"].update(explicit_metrics(scorer, split.test))
        per_user.to_csv(output / "per_user.csv", index=False)
        (output / "rankings.json").write_text(json.dumps(rankings))
    Path(args.output, "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
