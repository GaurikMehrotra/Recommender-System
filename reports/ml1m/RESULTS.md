# MovieLens 1M measured CPU benchmark

All eleven variants received four validation trials, a common 30-epoch neural cap
and patience four. Hyperparameters were frozen before one-time final full-ranking
evaluation at seeds 17, 42 and 73. Tables show mean ± sample standard deviation.
Deterministic classical fits have zero seed SD. This limited correlated finite grid
is a reproducible comparison under this budget, not exhaustive optimization.

Per-user leave-last-two-positive timestamp groups; ratings >=4. Train: 959,054
ratings, including 555,938 positives. Validation: 9,912 positives; test: 9,406 positives.
There are 6,033 eligible users and 3,883 catalog items; seven users are excluded.
Ranking excludes all train observations and test additionally excludes validation
positives. Chronology is per user, not global. Catalog metadata is transductive.
See MODEL_CARD.md for selection bias and candidate protocol.

## Accuracy

| Model | Recall@10 | NDCG@10 | MAP@10 | HitRate@10 |
|---|---|---|---|---|
| random | 0.0025 ± 0.0008 | 0.0013 ± 0.0004 | 0.0008 ± 0.0003 | 0.0036 ± 0.0013 |
| popularity | 0.0472 ± 0.0000 | 0.0263 ± 0.0000 | 0.0169 ± 0.0000 | 0.0681 ± 0.0000 |
| user_knn | 0.0812 ± 0.0000 | 0.0461 ± 0.0000 | 0.0302 ± 0.0000 | 0.1152 ± 0.0000 |
| item_knn | 0.0631 ± 0.0000 | 0.0372 ± 0.0000 | 0.0247 ± 0.0000 | 0.0928 ± 0.0000 |
| svd | 0.0888 ± 0.0007 | 0.0498 ± 0.0005 | 0.0327 ± 0.0005 | 0.1248 ± 0.0005 |
| mf | 0.0380 ± 0.0020 | 0.0202 ± 0.0012 | 0.0124 ± 0.0008 | 0.0545 ± 0.0025 |
| bpr | 0.0597 ± 0.0005 | 0.0342 ± 0.0002 | 0.0224 ± 0.0002 | 0.0869 ± 0.0007 |
| gmf | 0.0580 ± 0.0103 | 0.0337 ± 0.0069 | 0.0223 ± 0.0050 | 0.0858 ± 0.0160 |
| mlp | 0.0520 ± 0.0075 | 0.0291 ± 0.0052 | 0.0185 ± 0.0039 | 0.0772 ± 0.0100 |
| neumf | 0.0527 ± 0.0077 | 0.0291 ± 0.0047 | 0.0183 ± 0.0034 | 0.0783 ± 0.0106 |
| two_tower | 0.0417 ± 0.0031 | 0.0237 ± 0.0015 | 0.0145 ± 0.0009 | 0.0679 ± 0.0046 |

| Model | Recall@20 | NDCG@20 | MAP@20 | HitRate@20 |
|---|---|---|---|---|
| random | 0.0049 ± 0.0007 | 0.0020 ± 0.0004 | 0.0010 ± 0.0002 | 0.0074 ± 0.0018 |
| popularity | 0.0816 ± 0.0000 | 0.0361 ± 0.0000 | 0.0196 ± 0.0000 | 0.1139 ± 0.0000 |
| user_knn | 0.1366 ± 0.0000 | 0.0621 ± 0.0000 | 0.0347 ± 0.0000 | 0.1863 ± 0.0000 |
| item_knn | 0.1096 ± 0.0000 | 0.0505 ± 0.0000 | 0.0283 ± 0.0000 | 0.1561 ± 0.0000 |
| svd | 0.1442 ± 0.0008 | 0.0657 ± 0.0004 | 0.0370 ± 0.0005 | 0.1976 ± 0.0008 |
| mf | 0.0623 ± 0.0010 | 0.0272 ± 0.0009 | 0.0143 ± 0.0008 | 0.0866 ± 0.0007 |
| bpr | 0.0992 ± 0.0017 | 0.0455 ± 0.0005 | 0.0254 ± 0.0003 | 0.1397 ± 0.0021 |
| gmf | 0.0978 ± 0.0159 | 0.0449 ± 0.0083 | 0.0253 ± 0.0054 | 0.1370 ± 0.0222 |
| mlp | 0.0898 ± 0.0118 | 0.0399 ± 0.0064 | 0.0214 ± 0.0043 | 0.1271 ± 0.0168 |
| neumf | 0.0900 ± 0.0121 | 0.0396 ± 0.0059 | 0.0212 ± 0.0037 | 0.1280 ± 0.0175 |
| two_tower | 0.0690 ± 0.0042 | 0.0315 ± 0.0018 | 0.0166 ± 0.0010 | 0.1062 ± 0.0059 |

## Beyond accuracy at 20

| Model | coverage | novelty_bits | intra_list_diversity | head_share |
|---|---|---|---|---|
| random | 1.0000 ± 0.0000 | 14.1838 ± 0.0079 | 0.7797 ± 0.0007 | 0.1785 ± 0.0004 |
| popularity | 0.0695 ± 0.0000 | 8.3163 ± 0.0000 | 0.7280 ± 0.0000 | 1.0000 ± 0.0000 |
| user_knn | 0.2125 ± 0.0000 | 8.8996 ± 0.0000 | 0.6958 ± 0.0000 | 0.9961 ± 0.0000 |
| item_knn | 0.2094 ± 0.0000 | 8.7570 ± 0.0000 | 0.7168 ± 0.0000 | 0.9940 ± 0.0000 |
| svd | 0.2436 ± 0.0004 | 9.3019 ± 0.0023 | 0.6755 ± 0.0004 | 0.9944 ± 0.0001 |
| mf | 0.0535 ± 0.0006 | 8.5587 ± 0.0196 | 0.7287 ± 0.0065 | 0.9999 ± 0.0000 |
| bpr | 0.1786 ± 0.0013 | 8.5695 ± 0.0074 | 0.7106 ± 0.0029 | 0.9993 ± 0.0000 |
| gmf | 0.2131 ± 0.1192 | 8.6695 ± 0.2961 | 0.6991 ± 0.0181 | 0.9928 ± 0.0062 |
| mlp | 0.1545 ± 0.0763 | 8.5777 ± 0.1819 | 0.7039 ± 0.0106 | 0.9983 ± 0.0020 |
| neumf | 0.1428 ± 0.0693 | 8.5651 ± 0.1673 | 0.7098 ± 0.0091 | 0.9988 ± 0.0016 |
| two_tower | 0.2211 ± 0.0076 | 10.6684 ± 0.1342 | 0.5419 ± 0.0064 | 0.7066 ± 0.0324 |

Explicit MF RMSE: 0.8560 ± 0.0001;
MAE: 0.7194 ± 0.0001.
These are errors on held-out positive ratings, a biased explicit-rating sample.

## Prespecified comparison

BPR minus item-kNN NDCG@20: -0.00501; paired-user bootstrap
95% CI [-0.00772, -0.00220]. Users are paired after averaging
three seeds. This interval captures user sampling only, conditional on this dataset;
it ignores seed, temporal and dataset uncertainty and makes no causal claim.

## Diagnostics and hardware

Validation embedding ablations hold other trial-zero parameters fixed, varying
16/32/64/128 dimensions. Exact matching completed tuning fits are reused; selected
seed-42 neural fits are likewise reused for final evaluation, preserving independent
seed-17 and seed-73 fits. Fixed test reranking penalties 0.05/0.2 describe the
accuracy/popularity tradeoff; no penalty is selected on test and score scales differ.
Cold-start CSV uses the prespecified default 16-dimensional full config (not the
selected headline hyperparameters), validation only, one seed (42), disjoint cohorts excluded from
early stopping, full-catalog candidates, MF ridge fold-in for 1-5 revealed likes,
and content-aware two-tower inference. It is diagnostic, not a headline estimate.

CPU cloud Linux, four threads, Python 3.12, CPU PyTorch; GPU/Kaggle was not executed.
- tuning: 1036.1s, peak child-process RSS 749.2 MiB.
- ablations: 1774.0s, peak child-process RSS 726.4 MiB.
- cold: 104.8s, peak child-process RSS 708.7 MiB.
- final: 883.4s, peak child-process RSS 700.9 MiB.

100 warm two-tower retrieval requests: p50 1.181ms, p95 1.588ms;
process RSS 375.8 MiB; exported arrays 11.070 MiB.
Latency includes explanations and excludes HTTP transport. Measurements are on this
cloud CPU and do not guarantee equivalent performance on an Intel i3/8GB laptop.
Memory measurements establish <5GB for these executed commands, not every config.

Only aggregates/configurations are published; no data, user metrics, catalogs,
checkpoints, embeddings or logs. Source release tag: ml1m-benchmark-v1.

## Observed conclusion

svd has the highest mean NDCG@20 (0.0657) under this
fixed budget. This ranking does not establish universal superiority or statistical
significance for every pair. The paired interval above addresses only the
prespecified BPR/item-kNN comparison. Neural models are not assumed to win.
