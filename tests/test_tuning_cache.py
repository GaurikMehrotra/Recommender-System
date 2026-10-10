"""Cached validation results must match the entire experimental protocol."""

import json
from pathlib import Path

from src.tune import cached_trial


def test_cached_trial_requires_matching_config_and_data(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.tune.data_fingerprint", lambda _: "data-v1")
    base = {"data_path": "example", "epochs": 30}
    config = {"seed": 42, "params": {"embedding_dim": 16}}
    directory = tmp_path / "tuning" / "mf" / "0"
    directory.mkdir(parents=True)
    manifest = tmp_path / "TUNING_MANIFEST.json"
    manifest.write_text(json.dumps({"config": base, "data_fingerprint": "data-v1"}))
    assert cached_trial(base, tmp_path, "mf", config) is None
    (directory / "TRIAL_COMPLETED.json").write_text(json.dumps({"config": config}))
    assert cached_trial(base, tmp_path, "mf", config) == directory
    assert cached_trial(base, tmp_path, "mf", dict(config, seed=17)) is None
    assert cached_trial(dict(base, epochs=31), tmp_path, "mf", config) is None
    monkeypatch.setattr("src.tune.data_fingerprint", lambda _: "data-v2")
    assert cached_trial(base, tmp_path, "mf", config) is None
