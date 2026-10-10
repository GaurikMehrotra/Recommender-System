"""Configuration, deterministic execution, and portable checkpoint utilities."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    """Read a YAML configuration without executing constructors."""
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def seed_everything(seed: int, threads: int = 4) -> None:
    """Seed CPU/GPU RNGs and constrain CPU parallelism for laptops."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)


def device_for(requested: str = "auto") -> torch.device:
    """Auto-select GPU when available, while permitting an explicit CPU run."""
    return torch.device(
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested == "auto"
        else requested
    )


def atomic_json(path: str | Path, value: object) -> None:
    """Write complete JSON atomically so disconnected sessions leave no partial manifest."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def data_fingerprint(data_path: str | Path) -> str:
    """Hash ratings and item metadata to reject resume after silent data changes."""
    import hashlib

    path = Path(data_path)
    names = ["u.data", "u.item"] if (path / "u.data").exists() else ["ratings.dat", "movies.dat"]
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        with (path / name).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()
