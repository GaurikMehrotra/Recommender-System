"""Produce figures from measured data/results; no invented benchmark values."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.preprocess import load_movielens


def eda(data_path: str, output: Path) -> None:
    """Plot full-dataset descriptive statistics; never use EDA to tune on test."""
    ratings, items, _ = load_movielens(data_path)
    output.mkdir(parents=True, exist_ok=True)
    plots = {
        "popularity": ratings.groupby("item").size().sort_values(ascending=False).to_numpy(),
        "activity": ratings.groupby("user").size().to_numpy(),
        "ratings": ratings.rating.to_numpy(),
    }
    for name, values in plots.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        if name == "popularity":
            ax.plot(range(1, len(values) + 1), values)
            ax.set(xlabel="Item popularity rank", ylabel="Observed rating count", yscale="log")
        else:
            ax.hist(values, bins=5 if name == "ratings" else 30)
            ax.set(xlabel="Rating" if name == "ratings" else "Ratings per user", ylabel="Count")
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=150)
        plt.close(fig)
    sparsity = 1 - len(ratings) / (ratings.user.nunique() * len(items))
    (output / "eda.txt").write_text(
        f"Events: {len(ratings)}; users: {ratings.user.nunique()}; items: {len(items)}; sparsity: {sparsity:.6f}\n"
    )


def results(path: str, output: Path) -> None:
    """Plot mean and seed standard deviation of measured full-ranking NDCG."""
    frame = pd.read_csv(path)
    summary = frame.groupby("model")["NDCG@20"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(12, 5))
    summary["mean"].plot.bar(yerr=summary["std"].fillna(0), ax=ax)
    ax.set(ylabel="Full-ranking NDCG@20", title="Mean ± sample SD across training seeds")
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "ndcg20.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/ml-100k")
    parser.add_argument("--results")
    parser.add_argument("--output", default="reports/figures")
    args = parser.parse_args()
    if args.results:
        results(args.results, Path(args.output))
    else:
        eda(args.data, Path(args.output))


if __name__ == "__main__":
    main()
