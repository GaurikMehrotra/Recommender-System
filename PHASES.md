# Phase guide and terminal commands

Run from the repository root with an activated environment. Source files contain complete
implementations; no notebook hides reusable training/evaluation logic. These are the
commands for reproducing phases, not a claim every experiment has executed.

Because the latest request asks to continue through all phases, the initial delivery
is one reviewed implementation commit; the phase commit commands below are useful
when independently rebuilding the project. They only create a commit when files have
changed. GPU notebook cells for phases 6-10 are in [KAGGLE.md](KAGGLE.md).

## Phase 1: Environment and data (LOCAL)

Why: A reproducible environment and sparse storage make later comparisons practical.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -e .
python -m src.data.download --size 100k
```

Expected: Verified HTTPS archive, SHA256 provenance file and extracted ml-100k; loader creates CSR without a dense user-item matrix.

```bash
git add src app tests configs notebooks reports .github README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE
git commit -m "feat: add sparse data loading and reproducible environment"
```

## Phase 2: EDA (LOCAL)

Why: Descriptive plots expose sparsity and popularity imbalance.

```bash
python -m src.report --data data/raw/ml-100k
python -m pip install jupyterlab
jupyter lab notebooks/01_eda.ipynb
```

Expected: Three descriptive figures and measured sparsity; each notebook plot has a computed takeaway.

```bash
git add src app tests configs notebooks reports .github README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE
git commit -m "docs: add sparse MovieLens exploration"
```

## Phase 3: Splits and evaluation (LOCAL)

Why: Correct evaluation matters more than architecture complexity.

```bash
python -m pytest tests/test_protocol.py -q
```

Expected: Hand-computed metrics, strict timestamp groups, sparse handling and leakage tests pass.

```bash
git add src app tests configs notebooks reports .github README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE
git commit -m "feat: add chronological full-ranking evaluation"
```

## Phase 4: Baselines (LOCAL)

Why: Strong reference points prevent misleading neural comparisons.

```bash
python -m src.train --model random --output runs/debug/random
python -m src.train --model popularity --output runs/debug/popularity
```

Expected: Validation metrics.json; full catalog scored; test untouched.

```bash
git add src app tests configs notebooks reports .github README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE
git commit -m "feat: add random and popularity baselines"
```

## Phase 5: Classical collaborative filtering (LOCAL)

Why: Well-tuned classical CF is often competitive.

```bash
python -m src.train --model user_knn --output runs/debug/user_knn
python -m src.train --model item_knn --output runs/debug/item_knn
python -m src.train --model svd --output runs/debug/svd
```

Expected: Sparse pruned similarities and validation ranking metrics for each model.

```bash
git add src app tests configs notebooks reports .github README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE
git commit -m "feat: add sparse neighborhoods and truncated SVD"
```

## Phase 6: Biased matrix factorization (LOCAL debug, KAGGLE full)

Why: MF establishes a learned factor baseline with explicit ratings.

```bash
python -m src.train --model mf --output runs/debug/mf
python -m src.train --model mf --output runs/debug/mf --resume
python -m tensorboard.main --logdir runs/debug/mf/tensorboard
```

Expected: Epoch losses, validation NDCG, best.pt/last.pt and TensorBoard scalars; completed resume restores best model.

```bash
git add src app tests configs notebooks reports .github README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE
git commit -m "feat: add resumable biased matrix factorization"
```

## Phase 7: BPR (LOCAL debug, KAGGLE full)

Why: Pairwise loss directly trains relative preferences.

```bash
python -m src.train --model bpr --output runs/debug/bpr
```

Expected: Pairwise logistic loss and train-only unobserved negatives; validation early stopping.

```bash
git add src app tests configs notebooks reports .github README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE
git commit -m "feat: add Bayesian personalized ranking"
```

## Phase 8: Neural collaborative filtering (LOCAL debug, KAGGLE full)

Why: Separate variants reveal whether nonlinear interactions actually help.

```bash
python -m src.train --model gmf --output runs/debug/gmf
python -m src.train --model mlp --output runs/debug/mlp
python -m src.train --model neumf --output runs/debug/neumf
```

Expected: Three separate model variants, BCE losses, bounded item scoring and validation metrics.

```bash
git add src app tests configs notebooks reports .github README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE
git commit -m "feat: add GMF MLP and NeuMF"
```

## Phase 9: Content two-tower (LOCAL debug, KAGGLE full)

Why: Metadata can support unseen-item and short-history retrieval.

```bash
python -m src.train --model two_tower --output runs/debug/two_tower
```

Expected: Content-only item tower and history-feature user tower; no held-out history features.

```bash
git add src app tests configs notebooks reports .github README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE
git commit -m "feat: add content-aware two-tower retrieval"
```

## Phase 10: Fair tuning and headline evaluation (KAGGLE; 100K debugging LOCAL)

Why: Validation-only selection and multiple seeds reduce optimistic conclusions.

```bash
python -m src.tune --config configs/debug_cpu.yaml --stage tune --output runs/ml100k
python -m src.tune --config configs/debug_cpu.yaml --stage ablation --output runs/ml100k
python -m src.cold_start --config configs/cold_start.yaml --output runs/ml100k_cold
python -m src.tune --config configs/debug_cpu.yaml --stage final --output runs/ml100k
```

Expected: Frozen selection.yaml, >=3-seed results/summary CSV and prespecified bootstrap CI. Run full.yaml on Kaggle instead for headline 1M results.

```bash
git add src app tests configs notebooks reports .github README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE
git commit -m "feat: add equal-budget tuning and multi-seed reporting"
```

## Phase 11: Beyond accuracy and reranking (LOCAL)

Why: Accuracy alone hides concentration and catalog exposure.

```bash
python -m src.report --results runs/ml100k/results.csv
jupyter lab notebooks/02_results_analysis.ipynb
```

Expected: Measured coverage/novelty/diversity/head-share and fixed reranking curves, with seed SD.

```bash
git add src app tests configs notebooks reports .github README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE
git commit -m "feat: report recommendation quality beyond accuracy"
```

## Phase 12: Cold start and explainability (LOCAL)

Why: Artificial cold-start cohorts reveal assumptions hidden by warm-user evaluation.

```bash
python -m src.cold_start --output runs/debug/cold_start
jupyter lab notebooks/03_bias_and_cold_start.ipynb
```

Expected: Validation-only 1-5-history user simulations and disjoint held-item comparisons. Genre/year explanations disclose their provenance.

```bash
git add src app tests configs notebooks reports .github README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE
git commit -m "feat: add cold-start diagnostics and similarity explanations"
```

## Phase 13: Serving (LOCAL)

Why: Retrieval artifacts connect offline experiments to a working application.

```bash
python -m src.recommend --checkpoint runs/debug/two_tower/best.pt --artifacts artifacts/demo
python -m src.recommend --artifacts artifacts/demo --benchmark
python -m uvicorn src.api:create_app --factory --host 127.0.0.1 --port 8000
```

Expected: Exported embeddings, measured p50/p95/RSS, and functional HTTP recommendations; start demo in another terminal with python -m app.app.

```bash
git add src app tests configs notebooks reports .github README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE
git commit -m "feat: add retrieval API and interactive movie demo"
```

## Phase 14: Quality and CI (LOCAL)

Why: Synthetic CI checks behavior without redistributing data.

```bash
make lint test
```

Expected: Ruff check/format and synthetic pytest suite pass without downloading data.

```bash
git add src app tests configs notebooks reports .github README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE
git commit -m "ci: add CPU quality and functional checks"
```

## Phase 15: Documentation and reproducibility (LOCAL)

Why: Honest documentation distinguishes measured evidence from assumptions.

```bash
python -m src.report --results runs/ml100k/results.csv
git status --short
```

Expected: Publish measured aggregate figures only; README/model card explain protocol, uncertainty, bias and artifact restrictions.

```bash
git add src app tests configs notebooks reports .github README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE
git commit -m "docs: document benchmark reproduction and model limitations"
```

## Phase 16: Final verification and GitHub push (LOCAL)

Why: Verify the exact code and remote before calling delivery complete.

```bash
make lint test
git status --short
git add src app tests configs notebooks reports README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE .github
git commit -m "feat: build reproducible sparse recommender research pipeline"
git push -u origin HEAD:main
```

Expected: Push existing repo without force, then verify remote HEAD. Never add data/runs/artifacts.

```bash
git add src app tests configs notebooks reports .github README.md MODEL_CARD.md KAGGLE.md PHASES.md Makefile requirements.txt requirements-lock.txt requirements-cpu.lock pyproject.toml .gitignore LICENSE
git commit -m "release: verify recommender portfolio project"
```

For a **new, differently named** repository only (not this existing origin):

```bash
gh repo create YOUR_NEW_REPOSITORY_NAME --public --source=. --remote=publish --push
```

Authenticate GitHub CLI using its supported secure flow if you choose that optional route.
The cloud workflow uses existing proxy-backed native Git authentication.
