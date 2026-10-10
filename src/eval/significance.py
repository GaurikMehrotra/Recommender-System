"""Paired user bootstrap with training-seed averaging and explicit uncertainty limits."""

import numpy as np


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, draws: int = 2000, seed: int = 42
) -> dict[str, float]:
    """Estimate mean difference CI using paired users; not seed-level uncertainty.

    Input arrays are aligned per-user scores averaged across the SAME seeds.
    For multiple comparisons, prespecify a main comparison or adjust separately.
    """
    difference = np.asarray(a) - np.asarray(b)
    if difference.ndim != 1 or len(difference) == 0 or np.shape(a) != np.shape(b):
        raise ValueError("Need equally sized nonempty aligned per-user vectors")
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(difference, len(difference), replace=True).mean() for _ in range(draws)]
    )
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_difference": float(difference.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }
