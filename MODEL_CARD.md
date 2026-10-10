# Model card

## Status and scope
This is an educational MovieLens research pipeline, not a production Netflix model.
All required model families are implemented. Synthetic functional checks are distinct
from MovieLens effectiveness measurements. See `reports/VALIDATION.md` for actual
executed checks. Measured CPU benchmarks are linked from the README; Kaggle GPU execution is unrun.

## Data and license
MovieLens 100K is for debugging; MovieLens 1M is the main benchmark. Download directly
from GroupLens and read each archive's README and use conditions. This repository
contains no dataset, user records, checkpoints, embeddings, or training logs.
Code is MIT licensed; that does not relicense MovieLens. Cite:

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and
Context. ACM Transactions on Interactive Intelligent Systems 5(4), Article 19.
https://doi.org/10.1145/2827872

Official source: https://grouplens.org/datasets/movielens/

## Models and objectives
Random; positive-event popularity; sparse user/item cosine kNN with overlap shrinkage;
uncentered implicit TruncatedSVD; explicit MF with biases (MSE); BPR-MF (pairwise
logistic); GMF/MLP/NeuMF (binary cross entropy); content-only item two-tower with
history-pooled content users (binary cross entropy). Adam weight decay regularizes
neural parameters. MF scores are not clipped for RMSE. Optional EASE/25M are outside
this implementation's required CPU workflow.

## Evaluation and honest assumptions
Positives are ratings >=4 by default. Eligible users need three distinct positive
timestamps. Their penultimate positive timestamp group is validation; the last is
test. Train includes all ratings strictly before validation. Intermediate observations
are discarded, and events sharing held timestamps are handled as groups. Repeat
user-item events are rejected. IDs and the movie catalog are transductive metadata;
no fitted rating statistics use held-out interactions.

Strict chronology is per user, not global across users. This evaluates personalized
next-positive ranking within the observed catalog, not historical availability or a
causal deployment experiment. Low held-out ratings are not modeled as ranking truth.
MF RMSE/MAE are therefore on held-out **positive ratings**, not an unbiased sample
of all ratings. Users excluded by the protocol are reported by preparation code;
the benchmark should not generalize to those users without another protocol.

Full ranking covers all catalog items, including items with no train positives;
exclude every train-observed rating. Test additionally excludes validation positives.
No sampled-negative evaluation is reported. Batch scores use O(batch_users * items)
memory; never a full dense users-by-items matrix. Stable ties favor lower item IDs,
a reproducible convention that can matter for sparse baselines and untrained items.

Macro Recall/NDCG/MAP/HitRate at 10/20; MAP denominator is min(K, number relevant).
Beyond-accuracy uses add-one train-popularity novelty, catalog coverage, cosine
genre/year intra-list diversity, and share in the top 20% most popular catalog items.
Fixed reranking strengths produce a descriptive tradeoff curve, never a best test
lambda. Score-scale differences limit direct lambda comparisons between models.

Equal numbers of validation trials and the same epoch/patience cap are used for all
models. Compute costs differ; equal trials do not mean equal FLOPs. Random has no
quality hyperparameter and popularity's positive temperature is order-preserving;
these receive redundant control trials rather than lucky validation seed selection.
The shared finite grid is a limited search, not evidence of exhaustive optimal tuning.
MF's explicit objective differs from implicit losses; this is part of the comparison.

Headline runs require >=3 fixed seeds and report sample SD, not standard error. The
prespecified BPR vs item-kNN paired user bootstrap averages seeds first; its CI only
captures user-sampling variability conditional on this dataset/configuration. It
ignores training-seed and temporal uncertainty, and is not causal significance.

## Cold start and explanations
Cold-user/item simulations use validation only. Held users/items are removed from
training before fitting. Diagnostic cohorts are also excluded from early-stop validation. Cold items are excluded
from negative sampling. New users with >=5 available positives reveal earliest 1-5
train positives at inference;
MF fits a new user factor and bias by ridge regression on the revealed ratings,
and the content tower
uses its learned user network. New-item metrics condition on held-item validation positives while ranking the
full catalog for both models. Original train-history masks prevent repeat
recommendations without fitting withheld item feedback. MF's untrained item embeddings are a weak diagnostic;
content-only items can generalize from genres/year. These artificial simulations do
not prove performance for real new users/items. Fold-in regularization is prespecified, not chosen using diagnostic targets.
Default cold-start runs are one-seed
diagnostics; repeat seeds for any headline claims.

Explanations are "because you liked X" based on genre/year cosine similarity. They
are explicitly not causal attribution or proof that X drove the model score. Known-user exclusions include all train observations; explanations use only positive
training history. Selected-movie histories are explicitly likes.

## Fairness, intended use, and limitations
MovieLens volunteers and historical preferences are not representative of all
people or today's audiences. Popularity bias can obscure niche movies and minority
preferences. Genre diversity is not demographic fairness, and these measurements
cannot establish protected-group parity. User demographics are not used as features.
No age-appropriate filtering, safety review, production auth/rate limiting, online
A/B evaluation, deletion pipeline, drift detection, or real-time updates are provided.
Use for learning and offline research; do not deploy as consequential personalization.

## Checkpoint publishing plan
Keep raw data, ID mappings, embeddings, checkpoints, logs, and retrieval catalogs out
of Git. Do not publish MovieLens-derived artifacts until reviewing GroupLens terms
and obtaining any necessary permission. Publish code, configs, aggregate tables,
protocol, seeds, software versions, and provenance digests. If a permissible artifact
is released separately, include model version, config, checksum, dataset citation,
limitations, and a trusted download location. Only load your own trusted PyTorch
checkpoints (`torch.load(..., weights_only=False)` can execute pickle payloads).
