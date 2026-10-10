"""Implicit matrix factorization trained with Bayesian personalized ranking."""

from src.models.mf import MF


class BPR(MF):
    """Loss -log sigmoid(s(u,i+) - s(u,i-)) + L2; no explicit rating regression."""

    explicit = False
