"""Scientifically meaningful hand calculations and leakage checks."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.data.negatives import NegativeSampler
from src.data.preprocess import interaction_matrix, load_movielens
from src.data.splits import assert_no_leakage, leave_last_out
from src.eval.beyond_accuracy import beyond_accuracy
from src.eval.evaluate import evaluate
from src.eval.metrics import ranking_metrics
from src.eval.significance import paired_bootstrap
from src.models.baselines import Random
from src.models.knn import sparse_similarity


def test_metrics_hand_computed():
    # Ranking [irrelevant, relevant, relevant]; truth {2,3,9}; K=3.
    m = ranking_metrics(np.array([1, 2, 3]), {2, 3, 9}, 3)
    assert m["Recall@3"] == pytest.approx(2 / 3)
    assert m["MAP@3"] == pytest.approx((1 / 2 + 2 / 3) / 3)
    assert m["NDCG@3"] == pytest.approx((1 / np.log2(3) + 1 / 2) / (1 + 1 / np.log2(3) + 1 / 2))
    assert m["HitRate@3"] == 1
    assert ranking_metrics(np.array([1]), set(), 10)["MAP@10"] == 0
    assert ranking_metrics(np.array([2, 3]), {2, 3}, 20)["NDCG@20"] == 1


def test_split_strict_and_ties(tiny):
    events, _, _ = load_movielens(tiny["data_path"])
    events.loc[(events.user == 0) & (events.timestamp == 106), "timestamp"] = 107
    split = leave_last_out(events)
    assert_no_leakage(split)
    assert len(split.validation) > 0 and len(split.test) > 0
    assert len(split.test[split.test.user == 0]) == 2
    events = pd.concat([events, events.iloc[:1]])
    with pytest.raises(ValueError, match="Repeated"):
        leave_last_out(events)


def test_sparse_and_negatives(tiny):
    events, items, _ = load_movielens(tiny["data_path"])
    matrix = interaction_matrix(events, (8, len(items)))
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.nnz == len(events)
    similarity = sparse_similarity(matrix, 3)
    assert sparse.isspmatrix_csr(similarity)
    assert max(np.diff(similarity.indptr)) <= 3
    sampler = NegativeSampler(matrix, 42)
    users = np.tile(np.arange(8), 100)
    negatives = sampler.sample(users)
    assert np.asarray(matrix[users, negatives]).sum() == 0
    with pytest.raises(ValueError, match="no unobserved"):
        NegativeSampler(sparse.csr_matrix(np.ones((1, 3)))).sample(np.array([0]))


def test_full_ranking_excludes_history():
    class Fixed:
        def score(self, users):
            return np.broadcast_to(np.array([100, 90, 80, 70, 60], dtype=float), (len(users), 5))

    train = sparse.csr_matrix(([1], ([0], [0])), shape=(1, 5))
    val = sparse.csr_matrix(([1], ([0], [1])), shape=(1, 5))
    truth = pd.DataFrame({"user": [0], "item": [2]})
    metrics, _, recs = evaluate(Fixed(), truth, train, ks=[1, 3], extra_seen=val)
    assert recs[0] == [2, 3, 4]
    assert metrics["Recall@1"] == 1


def test_random_batch_independence():
    model = Random(seed=42).fit(sparse.csr_matrix((4, 30)))
    np.testing.assert_array_equal(
        model.score(np.array([2, 3])), model.score(np.array([0, 1, 2, 3]))[2:]
    )


def test_beyond_accuracy_and_paired_ci():
    metrics = beyond_accuracy({0: [0, 1], 1: [1, 2]}, np.array([3, 2, 1]), np.eye(3), k=2)
    assert metrics["coverage"] == 1
    assert metrics["intra_list_diversity"] == 1
    interval = paired_bootstrap(np.ones(4), np.zeros(4), draws=100)
    assert interval == {"mean_difference": 1.0, "ci_low": 1.0, "ci_high": 1.0}


def test_cold_items_never_sampled():
    matrix = sparse.csr_matrix(([1], ([0], [0])), shape=(1, 5))
    sampler = NegativeSampler(matrix, 42, excluded_items=[1, 2])
    negative = sampler.sample(np.zeros(100, dtype=int))
    assert set(negative) == {3, 4}


def test_one_million_loader_noncontiguous_movie_ids(tmp_path):
    (tmp_path / "ratings.dat").write_text("1::5::4::100\n2::9::5::101\n")
    (tmp_path / "movies.dat").write_text(
        "5::First (1999)::Action|Comedy\n9::Second (2000)::Drama\n"
    )
    events, items, features = load_movielens(tmp_path)
    assert events.item.tolist() == [0, 1]
    assert events.user.tolist() == [0, 1]
    assert features.shape == (2, 20)
    assert len(items) == 2


def test_download_checksum_and_repeatability(tmp_path):
    import hashlib
    import zipfile

    from src.data.download import download

    archive = tmp_path / "ml-100k.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("ml-100k/u.data", "fictional ratings")
        handle.writestr("ml-100k/u.item", "fictional movie metadata")
    with pytest.raises(ValueError, match="checksum"):
        download("100k", tmp_path, "0" * 64)
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    first = download("100k", tmp_path, expected)
    assert download("100k", tmp_path, expected) == first
    (first / "u.data").write_text("pre-existing edit")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        download("100k", tmp_path, expected)


def test_archive_path_traversal_rejected(tmp_path):
    import zipfile

    from src.data.download import download

    with zipfile.ZipFile(tmp_path / "ml-100k.zip", "w") as handle:
        handle.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        download("100k", tmp_path)
