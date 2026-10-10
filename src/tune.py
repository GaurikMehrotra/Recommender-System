"""Equal validation trial budgets, locked selections, then three-seed final reporting."""

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.preprocess import interaction_matrix
from src.eval.beyond_accuracy import beyond_accuracy
from src.eval.evaluate import evaluate, explicit_metrics
from src.eval.significance import paired_bootstrap
from src.train import MODELS, NEURAL, fit, prepare
from src.utils import atomic_json, data_fingerprint, load_config


def trial_config(base: dict, model: str, trial: int, ablation_dim: int | None = None) -> dict:
    """Deterministic common grid; each model receives the same number of trials.

    Random has no meaningful quality hyperparameter; its repeated trials are
    controls, not validation-selected lucky seeds. Popularity temperature is
    order-preserving, so its repeated trials are also redundant, disclosed controls.
    """
    c = copy.deepcopy(base)
    dims = base["search"]["embedding_dims"]
    dim = ablation_dim or dims[trial % len(dims)]
    c["params"] = {"embedding_dim": dim}
    if model.endswith("knn"):
        c["params"] = {
            "k": base["search"]["neighbors"][trial % len(base["search"]["neighbors"])],
            "shrinkage": base["search"]["shrinkage"][trial % len(base["search"]["shrinkage"])],
        }
    if model == "popularity":
        c["params"] = {"temperature": 1.0}
    if model == "random":
        c["params"] = {}
    c["learning_rate"] = base["search"]["learning_rates"][
        trial % len(base["search"]["learning_rates"])
    ]
    c["weight_decay"] = base["search"]["weight_decays"][
        trial % len(base["search"]["weight_decays"])
    ]
    return c


def tune(base: dict, output: Path, models: list[str], resume: bool = False) -> None:
    """Select entirely on validation, preserving every trial's configuration/results."""
    if (output / "TEST_CONSUMED").exists():
        raise ValueError("Test already consumed; do not tune this experiment again")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": base,
        "models": models,
        "data_fingerprint": data_fingerprint(base["data_path"]),
    }
    manifest_path = output / "TUNING_MANIFEST.json"
    if resume:
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume requires unchanged tuning configuration and model list")
    else:
        atomic_json(manifest_path, manifest)
    prepared = prepare(base)
    selections, trials = {}, []
    for model in models:
        best = -np.inf
        for trial in range(base["tuning_trials"]):
            config = trial_config(base, model, trial)
            directory = output / "tuning" / model / str(trial)
            completed = directory / "TRIAL_COMPLETED.json"
            if resume and completed.exists():
                record = json.loads(completed.read_text())
                if record["config"] != config:
                    raise ValueError("Cached trial configuration changed")
                metrics = record["validation"]
            else:
                _, metrics = fit(
                    config,
                    model,
                    directory,
                    resume=resume and (directory / "last.pt").exists(),
                    prepared=prepared,
                )
                atomic_json(completed, {"config": config, "validation": metrics})
            trials.append({"model": model, "trial": trial, "config": config, "validation": metrics})
            if metrics["NDCG@20"] > best:
                selections[model], best = config, metrics["NDCG@20"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "trials.json").write_text(json.dumps(trials, indent=2))
    (output / "selection.yaml").write_text(yaml.safe_dump(selections))


def cached_trial(base: dict, output: Path, model: str, config: dict) -> Path | None:
    """Reuse only completed validation fits with matching data and full config."""
    manifest = output / "TUNING_MANIFEST.json"
    if not manifest.exists():
        return None
    saved = json.loads(manifest.read_text())
    if saved["config"] != base or saved["data_fingerprint"] != data_fingerprint(base["data_path"]):
        return None
    for completed in sorted((output / "tuning" / model).glob("*/TRIAL_COMPLETED.json")):
        record = json.loads(completed.read_text())
        if record["config"] == config:
            return completed.parent
    return None


def final(base: dict, output: Path, resume: bool = False) -> None:
    """Run selected configs on three fixed seeds; include fixed reranking strengths.

    This operation opens a one-time test marker BEFORE evaluation. An interrupted
    run needs deliberate recovery without changing selections; do not delete the
    marker merely to retune. Bootstraps compare prespecified BPR vs item-kNN.
    """
    selections = yaml.safe_load((output / "selection.yaml").read_text())
    if len(base["seeds"]) < 3:
        raise ValueError("Headline final reporting requires at least three seeds")
    manifest = {
        "base_config": base,
        "selections": selections,
        "data_fingerprint": data_fingerprint(base["data_path"]),
    }
    manifest_path = output / "FINAL_MANIFEST.json"
    if resume:
        if (
            not (output / "TEST_CONSUMED").exists()
            or json.loads(manifest_path.read_text()) != manifest
        ):
            raise ValueError("Final resume requires unchanged frozen selection and configuration")
    else:
        with (output / "TEST_CONSUMED").open("x") as handle:
            handle.write("Selection frozen; do not tune after test inspection.\n")
        atomic_json(manifest_path, manifest)
    prepared = prepare(base)
    split, _, features, seen, positives, _ = prepared
    extra = interaction_matrix(split.validation, seen.shape)
    popularity = np.asarray(positives.sum(axis=0)).ravel()
    rows, user_scores = [], {}
    for model, selected in selections.items():
        user_scores[model] = []
        for seed in base["seeds"]:
            config = dict(selected, seed=seed)
            directory = output / "final" / model / str(seed)
            completed = directory / "FINAL_COMPLETED.json"
            if resume and completed.exists():
                rows.extend(json.loads(completed.read_text())["rows"])
                user_scores[model].append(
                    pd.read_csv(directory / "per_user.csv").set_index("user")["NDCG@20"]
                )
                continue
            first_row = len(rows)
            cached = cached_trial(base, output, model, config)
            if model in NEURAL and cached is not None and not (directory / "last.pt").exists():
                directory.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached / "last.pt", directory / "last.pt")
                atomic_json(directory / "REUSED_VALIDATION_FIT.json", {"source": str(cached)})
            scorer, validation = fit(
                config,
                model,
                directory,
                resume=(resume or cached is not None) and (directory / "last.pt").exists(),
                prepared=prepared,
            )
            metrics, per_user, rankings = evaluate(
                scorer, split.test, seen, base["ks"], base["eval_batch_size"], extra
            )
            if model == "mf":
                metrics.update(explicit_metrics(scorer, split.test))
            metrics.update(beyond_accuracy(rankings, popularity, features))
            rows.append({"model": model, "seed": seed, **metrics})
            per_user.to_csv(directory / "per_user.csv", index=False)
            (directory / "rankings.json").write_text(json.dumps(rankings))
            (directory / "metrics.json").write_text(
                json.dumps({"config": config, "validation": validation, "test": metrics}, indent=2)
            )
            user_scores[model].append(per_user.set_index("user")["NDCG@20"])
            # Predetermined descriptive tradeoff curve; never select lambda on test.
            for penalty in base["rerank_penalties"]:
                rr, _, recs = evaluate(
                    scorer,
                    split.test,
                    seen,
                    base["ks"],
                    base["eval_batch_size"],
                    extra,
                    novelty_penalty=penalty,
                    popularity=popularity,
                )
                rr.update(beyond_accuracy(recs, popularity, features))
                rows.append({"model": f"{model}_rerank_{penalty}", "seed": seed, **rr})
            atomic_json(completed, {"rows": rows[first_row:]})
    pd.DataFrame(rows).to_csv(output / "results.csv", index=False)
    result_frame = pd.DataFrame(rows)
    summary = result_frame.groupby("model").agg(
        {c: ["mean", "std"] for c in result_frame.columns if c not in ["model", "seed"]}
    )
    summary.to_csv(output / "summary.csv")
    if "bpr" in user_scores and "item_knn" in user_scores:
        a = pd.concat(user_scores["bpr"], axis=1).mean(axis=1)
        b = pd.concat(user_scores["item_knn"], axis=1).mean(axis=1)
        common = a.index.intersection(b.index)
        (output / "significance.json").write_text(
            json.dumps(
                paired_bootstrap(a.loc[common].to_numpy(), b.loc[common].to_numpy()), indent=2
            )
        )
    print(summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--output", default="runs/benchmark")
    parser.add_argument("--stage", choices=["tune", "final", "ablation"], default="tune")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue unchanged tuning/final runs after interruption",
    )
    args = parser.parse_args()
    base, output = load_config(args.config), Path(args.output)
    if args.stage == "tune":
        tune(base, output, args.models, args.resume)
    elif args.stage == "final":
        final(base, output, args.resume)
    else:
        if (output / "TEST_CONSUMED").exists():
            raise ValueError("Run ablations before consuming test")
        rows = []
        prepared = prepare(base)
        for model in [
            m for m in args.models if m in ["mf", "bpr", "gmf", "mlp", "neumf", "two_tower", "svd"]
        ]:
            for dim in base["search"]["embedding_dims"]:
                config = trial_config(base, model, 0, dim)
                cached = cached_trial(base, output, model, config)
                if cached is not None:
                    metrics = json.loads((cached / "TRIAL_COMPLETED.json").read_text())[
                        "validation"
                    ]
                else:
                    _, metrics = fit(
                        config, model, output / "ablation" / model / str(dim), prepared=prepared
                    )
                rows.append({"model": model, "embedding_dim": dim, **metrics})
        pd.DataFrame(rows).to_csv(output / "ablations.csv", index=False)


if __name__ == "__main__":
    main()
