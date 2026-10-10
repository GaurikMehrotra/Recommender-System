# Executed validation and phase status

The cloud instance was validated with Python 3.12.14 and CPU PyTorch 2.14.1+cpu.
A fresh virtual environment installed the complete `requirements-cpu.lock` with
`--require-hashes`; editable installation and `pip check` passed. The saved install
script was executed again successfully, including ruff check/format and **31 pytest
checks passing**. GitHub Actions is configured but no remote CI outcome is claimed.

## Functional evidence (fictional data only)

The CLI workflow was exercised on a deterministic synthetic catalog: 12 users,
40 movies, 120 fictional ratings. All 11 required model variants trained/scored.
Equal-budget two-trial tuning, validation embedding ablations, cold-user histories
1-5, held-item diagnostics, three-seed final evaluation (17/42/73), aggregate CSVs,
paired user bootstrap, fixed reranking curves, plots and resumable final reporting
completed. Saved data fingerprints/configuration manifests prevent recovery with
changed data or selections. Interrupted MF recovery matched uninterrupted CPU
scores exactly in its unit test. No synthetic metric is presented as MovieLens
accuracy or a scientifically meaningful model comparison.

Both live services were started and stopped during validation. FastAPI POST and
Gradio client `/recommend` each returned 10 recommendations with selected items
excluded; title/explanation rows were verified. The content two-tower retrieval
artifact was exported and used for these checks. Demo labels distinguish synthetic
from MovieLens data.

On **40 synthetic movies**, 100 warm retrieval requests measured p50
0.477 ms, p95 0.529 ms; process RSS
379.0 MiB and exported in-memory arrays
0.0171 MiB. These numbers include explanation construction,
exclude HTTP transport, and do not predict MovieLens/laptop latency or memory.
The <5 GB target has not been established on MovieLens 100K/1M or the user's laptop.

## Experimental phase status

| Phases | Implementation/delivery | Real-data execution |
|---|---|---|
| 1 | CPU environment, official downloader, sparse loader and configs | Dependencies verified; MovieLens download blocked |
| 2 | EDA notebook, figures generator and calculated takeaways | Synthetic generator exercised; MovieLens EDA unrun |
| 3-9 | Chronological protocol, metrics, baselines, classical and all neural models | Synthetic tests/CLI pass; MovieLens effectiveness unrun |
| 10 | Equal-budget tuning, ablations, >=3 seeds, frozen final stage and bootstrap | Synthetic workflow passes; headline ML-1M/Kaggle unrun |
| 11-12 | Beyond-accuracy/reranking, cold-start simulation and similarity explanations | Synthetic diagnostics pass; MovieLens analyses unrun |
| 13 | Retrieval export, FastAPI, Gradio and measured latency/RSS | Live synthetic serving verified; real-data serving unrun |
| 14-15 | 31-test suite, CPU CI, docs, model card, license and publishing plan | Local checks pass; remote CI not observed |
| 16 | Final checks and native Git push to existing GitHub origin | Remote push must be verified by delivery process |

## External blockers and next actions

The official HTTPS MovieLens request returned proxy **403 Forbidden**, including
through the supported escalation flow. `files.grouplens.org` was added to the
saved environment draft; saving a draft did not apply the runtime policy. Review
and save that network change in environment settings, then publish the environment
when ready. Retry `python -m src.data.download --size 100k` and `--size 1m` once live
access is available. No mirrors, TLS bypasses or fabricated results were used.

This instance is CPU-only and no Kaggle runtime/tool is attached. Execute the exact
[Kaggle cells](../KAGGLE.md) in an Internet-enabled GPU notebook on your account for
ML-1M headline experiments. Publish only measured aggregate tables/figures and
revise the README after those runs. Optional 25M/EASE are deliberately outside the
required implementation. No claim of complete experimental execution, published
cloud snapshot or restoration in a new task is made.
