# MovieLens 100K measured debug benchmark

These are measured **debug-budget** results, not the ML-1M headline experiment.
All eleven models received two validation trials and a three-epoch neural cap
(patience two). Selected configurations were frozen before test; final seeds were
17, 42 and 73. Values are means ± sample standard deviations across those seeds.
Classical deterministic models have zero seed SD. The small search/training budget
can undertrain neural models and does not establish universal model superiority.

Per-user strict chronological last-two-positive timestamp groups; ratings >=4;
937 eligible users, full catalog of 1,682 items. Train: 92,219 ratings, including
51,960 positives. Validation: 1,636 positives; test: 1,730 positives. Six users were
excluded for insufficient distinct positive timestamps. Candidate masks include
all train observations and, for test, validation positives. Chronology is not global
across users. The full catalog is transductive metadata. See MODEL_CARD.md.

## Accuracy

| Model | Recall@10 | NDCG@10 | MAP@10 | HitRate@10 |
|---|---|---|---|---|
| random | 0.0056 ± 0.0011 | 0.0027 ± 0.0004 | 0.0014 ± 0.0002 | 0.0103 ± 0.0012 |
| popularity | 0.0884 ± 0.0000 | 0.0540 ± 0.0000 | 0.0344 ± 0.0000 | 0.1473 ± 0.0000 |
| user_knn | 0.1447 ± 0.0000 | 0.0889 ± 0.0000 | 0.0565 ± 0.0000 | 0.2220 ± 0.0000 |
| item_knn | 0.1236 ± 0.0000 | 0.0789 ± 0.0000 | 0.0510 ± 0.0000 | 0.1996 ± 0.0000 |
| svd | 0.1446 ± 0.0053 | 0.0896 ± 0.0019 | 0.0578 ± 0.0018 | 0.2149 ± 0.0049 |
| mf | 0.0669 ± 0.0038 | 0.0403 ± 0.0016 | 0.0238 ± 0.0012 | 0.1245 ± 0.0016 |
| bpr | 0.0860 ± 0.0013 | 0.0536 ± 0.0007 | 0.0336 ± 0.0012 | 0.1473 ± 0.0028 |
| gmf | 0.0116 ± 0.0023 | 0.0065 ± 0.0008 | 0.0036 ± 0.0008 | 0.0235 ± 0.0056 |
| mlp | 0.0884 ± 0.0045 | 0.0544 ± 0.0020 | 0.0348 ± 0.0016 | 0.1459 ± 0.0059 |
| neumf | 0.0879 ± 0.0018 | 0.0540 ± 0.0005 | 0.0345 ± 0.0013 | 0.1448 ± 0.0031 |
| two_tower | 0.0296 ± 0.0035 | 0.0188 ± 0.0022 | 0.0110 ± 0.0017 | 0.0594 ± 0.0071 |

| Model | Recall@20 | NDCG@20 | MAP@20 | HitRate@20 |
|---|---|---|---|---|
| random | 0.0115 ± 0.0020 | 0.0044 ± 0.0005 | 0.0018 ± 0.0002 | 0.0210 ± 0.0006 |
| popularity | 0.1477 ± 0.0000 | 0.0717 ± 0.0000 | 0.0393 ± 0.0000 | 0.2241 ± 0.0000 |
| user_knn | 0.2145 ± 0.0000 | 0.1103 ± 0.0000 | 0.0634 ± 0.0000 | 0.2967 ± 0.0000 |
| item_knn | 0.1905 ± 0.0000 | 0.0989 ± 0.0000 | 0.0571 ± 0.0000 | 0.2753 ± 0.0000 |
| svd | 0.2365 ± 0.0039 | 0.1170 ± 0.0027 | 0.0666 ± 0.0023 | 0.3173 ± 0.0040 |
| mf | 0.1081 ± 0.0021 | 0.0527 ± 0.0003 | 0.0273 ± 0.0007 | 0.1743 ± 0.0043 |
| bpr | 0.1349 ± 0.0067 | 0.0680 ± 0.0022 | 0.0376 ± 0.0014 | 0.2095 ± 0.0126 |
| gmf | 0.0220 ± 0.0060 | 0.0094 ± 0.0018 | 0.0043 ± 0.0008 | 0.0413 ± 0.0119 |
| mlp | 0.1401 ± 0.0041 | 0.0697 ± 0.0012 | 0.0389 ± 0.0013 | 0.2174 ± 0.0076 |
| neumf | 0.1363 ± 0.0022 | 0.0683 ± 0.0013 | 0.0383 ± 0.0014 | 0.2110 ± 0.0037 |
| two_tower | 0.0452 ± 0.0059 | 0.0233 ± 0.0031 | 0.0122 ± 0.0019 | 0.0829 ± 0.0118 |

## Beyond accuracy at 20

| Model | coverage | novelty_bits | intra_list_diversity | head_share |
|---|---|---|---|---|
| random | 0.9998 ± 0.0003 | 12.5948 ± 0.0064 | 0.7684 ± 0.0018 | 0.1700 ± 0.0023 |
| popularity | 0.0874 ± 0.0000 | 7.7703 ± 0.0000 | 0.7205 ± 0.0000 | 1.0000 ± 0.0000 |
| user_knn | 0.2937 ± 0.0000 | 8.5535 ± 0.0000 | 0.7251 ± 0.0000 | 0.9458 ± 0.0000 |
| item_knn | 0.1623 ± 0.0000 | 8.0831 ± 0.0000 | 0.7219 ± 0.0000 | 0.9959 ± 0.0000 |
| svd | 0.2887 ± 0.0012 | 8.6372 ± 0.0057 | 0.7109 ± 0.0010 | 0.9686 ± 0.0011 |
| mf | 0.0737 ± 0.0031 | 8.0669 ± 0.0126 | 0.7101 ± 0.0091 | 0.9989 ± 0.0003 |
| bpr | 0.1007 ± 0.0024 | 7.8378 ± 0.0154 | 0.7257 ± 0.0074 | 1.0000 ± 0.0000 |
| gmf | 0.3591 ± 0.0333 | 11.8401 ± 0.6646 | 0.7691 ± 0.0087 | 0.2921 ± 0.1019 |
| mlp | 0.0890 ± 0.0024 | 7.8027 ± 0.0125 | 0.7203 ± 0.0008 | 1.0000 ± 0.0000 |
| neumf | 0.0886 ± 0.0030 | 7.8118 ± 0.0091 | 0.7176 ± 0.0035 | 1.0000 ± 0.0000 |
| two_tower | 0.0632 ± 0.0116 | 9.6713 ± 0.1789 | 0.5614 ± 0.0234 | 0.6774 ± 0.0257 |

Explicit MF RMSE: 0.9605 ± 0.0089;
MAE: 0.8311 ± 0.0087.
These errors are on held-out positive ratings and have the disclosed selection bias.

## Prespecified comparison and diagnostics

BPR minus item-kNN mean NDCG@20 difference: -0.03088; paired-user
bootstrap 95% CI [-0.04019, -0.02192]. Users are paired after
averaging the same three seeds. This interval is conditional on this dataset and
ignores training-seed/dataset/temporal uncertainty. It is not a causal superiority
claim or a corrected family of multiple comparisons.

The aggregate CSV includes fixed reranking strengths 0.05 and 0.2 for every model.
These form a descriptive tradeoff curve; no strength was selected on test. Scores
have different scales across models, limiting cross-model lambda comparisons.
Validation-only embedding ablations are in ablations.csv. Cold-start diagnostics
use their own saved config, disjoint user/item cohorts, and one seed (42); they are
not headline estimates. Cold MF uses ridge fold-in on revealed ratings and content
users pool revealed movie features; early stopping excludes diagnostic cohorts.

No dataset rows, per-user metrics, recommendation lists, movies catalog, checkpoints,
embeddings or logs are included. Files here contain configurations and aggregates.
The source snapshot for the warm debug benchmark is Git tag ml100k-debug-v1.
Cold diagnostics were subsequently corrected to rank the full catalog, conditioning
truth on cold-item validation positives and excluding original historical observations;
the final source includes a regression check. Their saved aggregate CSV reflects
that corrected validation-only rerun. No warm test evaluation was repeated.
