"""All model families, resume behavior, export, and functional API checks."""

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.api import create_app
from src.recommend import Recommender, export
from src.train import MODELS, fit, prepare


@pytest.mark.parametrize("name", MODELS)
def test_end_to_end_all_models(tiny, tmp_path, name):
    prepared = prepare(tiny)
    scorer, metrics = fit(tiny, name, tmp_path / name, prepared=prepared)
    scores = scorer.score(np.array([0, 1]))
    assert scores.shape == (2, 32)
    assert np.isfinite(scores).all()
    assert 0 <= metrics["NDCG@20"] <= 1
    if hasattr(scorer, "model"):
        assert scorer.model(torch.tensor([0, 1]), torch.tensor([2, 3])).shape == (2,)


def test_resume_completed_run(tiny, tmp_path):
    original, _ = fit(tiny, "bpr", tmp_path / "bpr")
    resumed, _ = fit(tiny, "bpr", tmp_path / "bpr", resume=True)
    np.testing.assert_array_equal(original.score(np.array([0])), resumed.score(np.array([0])))


@pytest.mark.parametrize("name", ["mf", "bpr", "two_tower"])
def test_export_and_api(tiny, tmp_path, name):
    fit(tiny, name, tmp_path / name)
    artifacts = tmp_path / "artifacts"
    export(tmp_path / name / "best.pt", artifacts)
    model = Recommender(artifacts)
    recs = model.recommend(liked=[0, 1], k=3)
    assert len(recs) == 3 and not {0, 1} & {r["item"] for r in recs}
    assert model.benchmark(3)["p95_ms"] >= 0
    with TestClient(create_app(str(artifacts))) as client:
        assert client.get("/health").json()["catalog_size"] == 32
        response = client.post("/recommend", json={"liked": [0, 1], "k": 3})
        assert response.status_code == 200
        assert len(response.json()) == 3
        assert client.post("/recommend", json={"user": 999}).status_code == 422


def test_cold_start_validation_only(tiny, tmp_path):
    import json

    import pandas as pd

    from src.cold_start import run

    run(tiny, tmp_path / "cold")
    results = pd.read_csv(tmp_path / "cold" / "cold_start.csv")
    assert set(results[results["mode"] == "users"].history_size) == {1, 2, 3, 4, 5}
    protocol = json.loads((tmp_path / "cold" / "protocol.json").read_text())
    state = torch.load(tmp_path / "cold" / "items" / "two_tower" / "last.pt", weights_only=False)
    assert state["config"]["excluded_training_items"] == protocol["held_items"]


def test_tuning_final_and_test_guard(tiny, tmp_path):
    import pandas as pd

    from src.tune import final, tune

    output = tmp_path / "experiment"
    tune(tiny, output, ["random", "item_knn", "bpr", "mf"])
    final(tiny, output)
    table = pd.read_csv(output / "results.csv")
    assert set(table.seed) == set(tiny["seeds"])
    assert (output / "significance.json").exists()
    assert "RMSE" in pd.read_csv(output / "summary.csv", header=[0, 1]).columns.get_level_values(0)
    with pytest.raises(FileExistsError):
        final(tiny, output)
    with pytest.raises(ValueError, match="Test already"):
        tune(tiny, output, ["random"])


def test_gradio_function(tiny, tmp_path):
    from app.app import build_demo

    fit(tiny, "two_tower", tmp_path / "two_tower")
    export(tmp_path / "two_tower" / "best.pt", tmp_path / "artifacts")
    demo = build_demo(str(tmp_path / "artifacts"))
    assert len(demo.fn([0, 1])) > 0


def test_interrupted_resume_matches_uninterrupted(tiny, tmp_path, monkeypatch):
    import src.train as train_module

    reference, _ = fit(tiny, "mf", tmp_path / "reference")
    actual_evaluate = train_module.evaluate
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Simulated disconnect after first saved epoch")
        return actual_evaluate(*args, **kwargs)

    monkeypatch.setattr(train_module, "evaluate", interrupted)
    with pytest.raises(RuntimeError, match="disconnect"):
        fit(tiny, "mf", tmp_path / "interrupted")
    monkeypatch.setattr(train_module, "evaluate", actual_evaluate)
    resumed, _ = fit(tiny, "mf", tmp_path / "interrupted", resume=True)
    np.testing.assert_allclose(
        reference.score(np.array([0, 1])), resumed.score(np.array([0, 1])), rtol=0, atol=0
    )


def test_final_resume_preserves_results_and_frozen_selection(tiny, tmp_path):
    import pandas as pd

    from src.tune import final, tune

    output = tmp_path / "resume_final"
    tune(tiny, output, ["item_knn", "bpr"])
    tune(tiny, output, ["item_knn", "bpr"], resume=True)
    final(tiny, output)
    original = pd.read_csv(output / "results.csv")
    final(tiny, output, resume=True)
    pd.testing.assert_frame_equal(original, pd.read_csv(output / "results.csv"))
    with pytest.raises(ValueError, match="unchanged"):
        final(dict(tiny, seeds=[1, 2, 3]), output, resume=True)


def test_resume_rejects_changed_data(tiny, tmp_path):
    from pathlib import Path

    fit(tiny, "mf", tmp_path / "mf")
    data = Path(tiny["data_path"]) / "u.data"
    rows = [row.split("\t") for row in data.read_text().splitlines()]
    rows[0][2] = "5"
    data.write_text("\n".join("\t".join(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="Resume requires"):
        fit(tiny, "mf", tmp_path / "mf", resume=True)
