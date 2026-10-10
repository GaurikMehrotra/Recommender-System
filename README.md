# MovieLens recommender system

A sparse, laptop-friendly research pipeline comparing classical collaborative
filtering with neural recommendation. The goal is a fair **measured** comparison,
not a presumption that deeper models win. Complete source lives under `src/`;
notebooks only explore data and analyze saved results.

**Current evidence:** synthetic model/training/serving checks are available in
[reports/VALIDATION.md](reports/VALIDATION.md). MovieLens download is currently
blocked in the onboarding runtime by its network policy. Consequently there are
**no measured MovieLens results yet**, and phases requiring actual MovieLens/Kaggle
experiments are not experimentally complete. The code and exact execution commands
are provided; no placeholder or published performance is substituted for measurements.

## Quick start (LOCAL, Linux/macOS terminal or VS Code)

Python 3.10+; validated cloud interpreter is Python 3.12. Windows users can use WSL.
The repository already exists; clone it rather than calling `gh repo create` again.

```bash
git clone https://github.com/GaurikMehrotra/Recommender-System.git
cd Recommender-System
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -e .
make lint test
python -m src.data.download --size 100k
python -m src.train --config configs/debug_cpu.yaml --model mf --output runs/debug/mf
python -m src.recommend --checkpoint runs/debug/mf/best.pt --artifacts artifacts/demo
make serve
```

In a second activated terminal:

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/recommend -H 'Content-Type: application/json' -d '{"liked":[0,1],"k":10}'
python -m app.app --artifacts artifacts/demo
```

API uses zero-based internal catalog/user IDs, not original MovieLens IDs. Selected
movies in the demo use catalog titles. A known user request uses learned embeddings;
new users of MF/BPR use a disclosed genre/year fallback. For learned new-user content
scoring, export a two-tower checkpoint. NCF is nonlinear pair scoring and is evaluated
in item blocks; the demo intentionally serves exportable retrieval models.

For local notebooks, install JupyterLab separately if desired (`pip install jupyterlab`),
then `jupyter lab`. Notebooks locate the checkout from root or `notebooks/`.

## Offline functional check when MovieLens access is unavailable

This generates **fictional** data and exercises the implementation, not effectiveness:

```bash
python -m src.data.synthetic --output data/synthetic/ml-100k
python -m src.train --config configs/synthetic_cpu.yaml --model two_tower --output runs/synthetic_debug
python -m src.recommend --checkpoint runs/synthetic_debug/best.pt --artifacts artifacts/synthetic
RECOMMENDER_ARTIFACTS=artifacts/synthetic python -m uvicorn src.api:create_app --factory
python -m app.app --artifacts artifacts/synthetic
```

The generator refuses overwriting existing data. API/demo commands run in separate
terminals; the synthetic demo is clearly labeled. Synthetic diagnostics cannot replace
MovieLens evaluation or establish the laptop memory target.

## Evaluation protocol

Ratings >=4 are implicit positives. For users with three distinct positive timestamp
groups, the last group is test and the preceding group validation; train stops strictly
before validation. Tied timestamps are held together. Train contains earlier low ratings
for explicit MF and exclusion, but implicit models fit only positives. Repeat pairs
are rejected. Chronology is per user; other users can train on later calendar dates.
This is not a globally temporal deployment benchmark. The full catalog is assumed
available as metadata. See [MODEL_CARD.md](MODEL_CARD.md) for selection biases.

Validation determines hyperparameters and early stopping. `src.tune --stage final`
freezes selections and creates an exclusive `TEST_CONSUMED` marker before scoring
three seeds. Do not inspect test results and retune. Recover an interrupted final run with the exact command plus `--resume`; manifests
require unchanged selections/configuration and reuse completed model-seed results. Run ablations
and cold-start diagnostics on validation **before** final evaluation.

Every ranking considers **all catalog items** and excludes prior observed items;
test also excludes validation positives. Small sampled-negative sets can disproportionately
reward models that separate easy random negatives, changing model ordering and inflating
accuracy compared with real retrieval. We never use them for reported ranking metrics.
Uniform negative sampling is used only to train implicit models.

Report macro Recall@10/20, NDCG@10/20, MAP@10/20, HitRate@10/20; explicit MF adds
RMSE/MAE on held-out positive ratings (selection bias disclosed). MAP normalizes by
min(K, number relevant). Coverage, smoothed novelty, content cosine diversity, and
popular-head share accompany accuracy. Full-ranking score allocations are bounded
by user batches; sparse interactions and blockwise sparse similarities avoid a dense
users-by-items matrix. Default four CPU threads and small embeddings target an 8 GB
laptop; <5 GB is a target to measure, not a universal proven guarantee.

## Models and fair tuning

Random and popularity; sparse user-kNN/item-kNN with cosine overlap shrinkage;
TruncatedSVD; PyTorch biased MF; BPR; GMF, MLP, NeuMF; content two-tower using genres
and year with history-pooled users. All neural models support CPU/auto GPU, validation
NDCG early stopping, TensorBoard, L2, and resumable RNG/optimizer checkpoints.
`configs/debug_cpu.yaml` uses 100K/3 epochs/16 dimensions. `configs/full.yaml` and the
model configs use 1M/30-epoch cap/auto device. No dense EASE or optional 25M is required.

The same trial count is allocated to every model, using a disclosed finite grid.
Random and popularity have no meaningful order-changing quality hyperparameters;
their duplicate control trials are disclosed rather than picking lucky seeds.
Epoch caps are equal, compute costs are not. Do not claim the small debug search
establishes optimality. MF can outperform NCF; conclusions must follow actual runs.

## Results

| Dataset | Status | Main accuracy table |
|---|---|---|
| Synthetic fixture | Functional validation only | Not a performance benchmark |
| MovieLens 100K | Download blocked in current cloud runtime | Not measured |
| MovieLens 1M (headline, >=3 seeds) | Not run | Not measured |

Generate aggregate tables with `src.tune` and figures with `src.report`; update this
table only with measured results and link the run provenance. Final output contains
per-seed `results.csv`, `summary.csv` (mean/sample SD), per-user scores, saved configs,
and a prespecified paired bootstrap BPR-minus-item-kNN NDCG@20 CI. The user bootstrap
conditions on averaged seeds and excludes seed/dataset/temporal uncertainty. Do not
present it as universal or causal superiority.

## All phases and exact terminal commands

[PHASES.md](PHASES.md) contains local commands, expected outcomes, and commit commands
for phases 1-16. Complete code is in the linked repository files rather than duplicating
thousands of lines in chat. [KAGGLE.md](KAGGLE.md) gives exact GPU notebook cells for
clone/install, full tuning, embedding ablations, three-seed final evaluation, resumable
training, and downloading results. GPU experiments have not been executed here.

## Data, reproducibility, and publication

MovieLens is obtained from https://grouplens.org/datasets/movielens/ and is not
redistributed. Read its terms. Cite Harper and Konstan (2015), *The MovieLens Datasets:
History and Context*, ACM TiiS 5(4), Article 19, https://doi.org/10.1145/2827872.
Downloads use verified HTTPS; recorded SHA256 is provenance, not an independent
trusted checksum. Supply `--sha256` if you have an authoritative expected digest.

For a hash-verified **Python 3.12 CPU** install matching the isolated validation:

```bash
python -m pip install --require-hashes -r requirements-cpu.lock --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps --no-build-isolation -e .
```

`requirements-cpu.lock` pins all resolved dependencies with hashes, including the CPU
PyTorch wheel; do not use that CPU lock in a Kaggle GPU session. It targets Python
3.12; use compatibility ranges for other Python versions and record your resolution.
`requirements.txt` declares compatibility ranges; `requirements-lock.txt` records
exact direct dependency versions from the validated Python 3.12 CPU environment.
Use `pip install -r requirements-lock.txt -e .` after installing the desired CPU/GPU
PyTorch build. It is not a hash-locked transitive dependency manifest or a promise of
bitwise equivalence across CUDA/BLAS platforms. Keep saved YAML/seeds, archive digest,
run config and software/hardware information with every benchmark.

Data, runs, checkpoints and exports are ignored. Only aggregate plots/tables may be
published after reviewing dataset terms. See the checkpoint plan and limitations in
[MODEL_CARD.md](MODEL_CARD.md). GitHub CI uses synthetic data only, CPU ruff + pytest.
