# Exact Kaggle notebook cells (GPU phases 6-10)

Enable **Internet** and select a free GPU accelerator in notebook settings. These
cells are supplied for execution on your Kaggle account; they have not run in this
CPU-only onboarding task. Total time depends on hardware and the search; no 30-hour
budget guarantee is claimed. Start on 100K before spending quota on 1M. No Kaggle
credentials are required by repository code. For a private repo use Kaggle secrets
securely and do not print or embed tokens.

Cell 1 — clone the existing public GitHub repository:

```python
from pathlib import Path
import subprocess
repo = Path('/kaggle/working/Recommender-System')
if not repo.exists():
    subprocess.run(['git','clone','https://github.com/GaurikMehrotra/Recommender-System.git',str(repo)],check=True)
%cd /kaggle/working/Recommender-System
```

Cell 2 — install while keeping Kaggle's CUDA PyTorch (do not use CPU index here):

```python
import sys, subprocess, torch

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-e", "."], check=True
)
print("GPU available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "Enable GPU before starting full experiments"
subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
```

Cell 3 — fetch official data; inspect printed digest for provenance:

```python
subprocess.run([sys.executable, "-m", "src.data.download", "--size", "1m"], check=True)
print(Path("data/raw/ml-1m.zip.sha256").read_text())
```

Cell 4 — debug individual GPU models on 100K before full runs:

```python
import yaml

subprocess.run([sys.executable, "-m", "src.data.download", "--size", "100k"], check=True)
config = yaml.safe_load(Path("configs/debug_cpu.yaml").read_text())
config["device"] = "auto"
Path("/kaggle/working/debug_gpu.yaml").write_text(yaml.safe_dump(config))
for model in ["mf", "bpr", "gmf", "mlp", "neumf", "two_tower"]:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--config",
            "/kaggle/working/debug_gpu.yaml",
            "--model",
            model,
            "--output",
            f"runs/gpu_debug/{model}",
        ],
        check=True,
    )
```

Cell 5 — phase 10, fair fixed-budget validation tuning (ALL models):

```python
subprocess.run(
    [
        sys.executable,
        "-m",
        "src.tune",
        "--config",
        "configs/full.yaml",
        "--stage",
        "tune",
        "--output",
        "runs/ml1m",
    ],
    check=True,
)
```

Cell 6 — validation-only embedding ablations and cold-start diagnostics, before test:

```python
subprocess.run(
    [
        sys.executable,
        "-m",
        "src.tune",
        "--config",
        "configs/full.yaml",
        "--stage",
        "ablation",
        "--output",
        "runs/ml1m",
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-m",
        "src.cold_start",
        "--config",
        "configs/full.yaml",
        "--output",
        "runs/ml1m_cold",
    ],
    check=True,
)
```

Cell 7 — one-time final test, three seeds and predefined reranking curve:

```python
subprocess.run(
    [
        sys.executable,
        "-m",
        "src.tune",
        "--config",
        "configs/full.yaml",
        "--stage",
        "final",
        "--output",
        "runs/ml1m",
    ],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "src.report", "--results", "runs/ml1m/results.csv"], check=True
)
```

Do not run Cell 7 twice or tune after inspecting test results. `TEST_CONSUMED` blocks
accidental repetition. Read `summary.csv`/`significance.json` and report what they say,
even when classical models beat NCF.

Cell 8 — resumable standalone training (same config, path, model):

```python
# Run this only after interruption of the corresponding standalone train command.
# An already-completed run simply restores its best checkpoint.
subprocess.run(
    [
        sys.executable,
        "-m",
        "src.train",
        "--config",
        "configs/mf.yaml",
        "--model",
        "mf",
        "--output",
        "runs/standalone_mf",
        "--resume",
    ],
    check=True,
)
```

Initially launch that standalone run without `--resume`. Checkpoints include epoch,
optimizer, best weights, early-stop state and sampler/torch RNGs. Kaggle disconnects
can erase unsaved working state: use Save Version with outputs and retain your own
permitted artifacts. Resuming requires restoring the whole run directory and exact
config/data. Recover interrupted tuning with the exact same command plus `--resume`:

```python
subprocess.run(
    [
        sys.executable,
        "-m",
        "src.tune",
        "--config",
        "configs/full.yaml",
        "--stage",
        "tune",
        "--output",
        "runs/ml1m",
        "--resume",
    ],
    check=True,
)
```

Recover an interrupted final phase with frozen selections and unchanged configuration:

```python
subprocess.run(
    [
        sys.executable,
        "-m",
        "src.tune",
        "--config",
        "configs/full.yaml",
        "--stage",
        "final",
        "--output",
        "runs/ml1m",
        "--resume",
    ],
    check=True,
)
```

Manifests reject changed configurations. Completed trials/model-seeds are reused;
unfinished neural fits resume from atomic optimizer/RNG checkpoints. Keep the entire
run directory. Never remove the test marker to perform a new search. An evaluation
interrupted mid-seed may repeat that seed with unchanged weights and protocol; this
is recovery within the same frozen final evaluation, not another tuning opportunity.


Cell 9 — download aggregate experiment outputs (no dataset/checkpoint redistribution):

```python
import zipfile
from IPython.display import FileLink, display

archive = Path("/kaggle/working/recommender_reports.zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for root in [Path("runs/ml1m"), Path("runs/ml1m_cold"), Path("reports")]:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".csv", ".json", ".yaml", ".png", ".md", ".txt"}:
                z.write(path, str(path))
display(FileLink(str(archive)))
```

Keep per-user CSVs private; publish only aggregate results after reviewing terms.
Never push `data/`, `runs/`, checkpoint files or exported embeddings/catalogs. Use
GitHub as the code/config bridge, not storage for MovieLens or sensitive artifacts.
