# Executed validation and phase status

Python 3.12.14, CPU PyTorch 2.14.1+cpu, Linux cloud CPU with four allocated
threads. A fresh isolated environment installed all dependencies from
requirements-cpu.lock with --require-hashes. Editable installation, pip check,
ruff check/format and all **33 pytest checks** passed. Tests include strict
chronological leakage, hand-computed metrics, sparse matrix handling, negative
sampling, model shapes, exact NCF scoring, ridge fold-in, checkpoint recovery,
cache guards, cold-start full-catalog candidates and a tiny end-to-end workflow.
Remote GitHub Actions is configured; its outcome was not independently observed.

## Phase evidence

| Phases | Delivered and executed |
|---|---|
| 1 | CPU hash-locked environment; official verified-HTTPS downloads of MovieLens 100K and 1M; sparse loaders and seed/config utilities |
| 2 | Real-data EDA figures and notebook execution with calculated takeaways |
| 3 | Per-user strict chronological train/validation/test protocol; leakage tests; batched full-ranking metrics |
| 4-9 | All eleven required model variants trained and evaluated on both real datasets; CPU/CUDA selection supported |
| 10 | Equal-budget tuning, validation embedding ablations, frozen three-seed final metrics and prespecified paired-user bootstrap |
| 11 | Coverage, novelty, genre/year diversity, popular-head share, fixed reranking tradeoff curves |
| 12 | Validation-only cold users with 1-5 revealed likes and withheld items; full-catalog ranking; similarity explanations |
| 13 | Real-data two-tower export; live FastAPI POST and Gradio named API each return ten recommendations excluding selected movies; measured latency/RSS |
| 14 | 33-test suite, Makefile, CPU GitHub Actions, anchored data/run/artifact ignore rules and MIT code license |
| 15 | README measured results/figures, model card, dataset citation and checkpoint publishing plan |
| 16 | Checks rerun and code/aggregate reports pushed to the existing GitHub main branch |

The synthetic 12-user/40-item fixture also exercises all model variants and the
complete tuning, ablation, cold-start, final/resume and serving workflow; it is
functional evidence only and is not presented as MovieLens effectiveness.
Interrupted CPU MF recovery matches uninterrupted scores in its unit test.
All three analysis notebooks executed successfully against real 100K outputs;
executed notebooks remain ignored to avoid embedding dataset records in Git.

## Measured benchmarks and limits

See [100K debug](ml100k/RESULTS.md) and [1M headline](ml1m/RESULTS.md) for actual
means, sample standard deviations, saved configurations, resource measurements,
validation diagnostics and bootstrap intervals. Neural models have a 3-epoch
100K debug cap or 30-epoch 1M cap; the finite correlated search is limited.
No data, per-user rows, catalogs, model weights or logs are committed.

The official GroupLens downloads now succeed. The earlier proxy 403 was resolved;
archive provenance digests are in DATA_PROVENANCE.json. No mirrors or TLS bypasses
were used. GPU/Kaggle execution is **unrun** because this instance has no CUDA or
attached Kaggle runtime. KAGGLE.md supplies exact notebook cells. The measured
headline experiment was executed on the cloud CPU; it is not mislabeled as GPU.
Optional 25M and EASE are omitted. Actual Intel i3 laptop timing and memory have
not been observed; cloud process RSS measurements do not guarantee every setting.

The tested reusable install_script and startup instructions are saved in the
cloud configuration draft. Saving does not publish the environment or establish
restoration in a fresh task; review/save and Publish remain environment UI actions.
