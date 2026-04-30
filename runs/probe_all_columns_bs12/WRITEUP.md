# Latin Square Ensemble Experiment

## What Was Run

This run scans the Latin-square ensemble over `K in {16, 64, 128}`, submodel budgets `{1M, 10M}`, and both min-entropy and max-entropy schedulers. The data is Wikitext-103 raw text tokenized with a 32,000-token BPE vocabulary and sequence length 128.

The architecture used here activates only a quarter of the Latin-square columns on each forward pass. For example, `K=128` evaluates 32 ensemble paths per batch instead of all 128. Inactive columns are sampled on later forwards, so the run keeps the global Latin-square structure while reducing per-step compute and memory.

Training uses AdamW with weight decay 0.1. The run is deliberately capped at a comparable early-convergence budget rather than full Wikitext epochs: the training curves from the previous pilot showed that most loss reduction happened by about 256 optimizer updates. The exact batch caps, effective batch sizes, and token exposure are recorded below.

## Run Scope

| k | total_params | scheduler | max_train_batches | max_eval_batches | effective_train_batch_size | effective_eval_batch_size | active_column_fraction | active_column_count | train_sequences_seen | train_tokens_seen | eval_sequences_seen | eval_tokens_seen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 1000000 | max | 1 | 1 | 12 | 12 | 1.00 | 16 | 24 | 3072 | 12 | 1536 |
| 64 | 1000000 | max | 1 | 1 | 12 | 12 | 1.00 | 64 | 24 | 3072 | 12 | 1536 |
| 16 | 1000000 | min | 1 | 1 | 12 | 12 | 1.00 | 16 | 24 | 3072 | 12 | 1536 |
| 64 | 1000000 | min | 1 | 1 | 12 | 12 | 1.00 | 64 | 24 | 3072 | 12 | 1536 |
| 16 | 10000000 | max | 1 | 1 | 12 | 12 | 1.00 | 16 | 24 | 3072 | 12 | 1536 |
| 16 | 10000000 | min | 1 | 1 | 12 | 12 | 1.00 | 16 | 24 | 3072 | 12 | 1536 |

## Results Table

| k | total_params | scheduler | d_model | ensemble_loss | avg_individual_loss | ensemble_gain | member_certainty_mean | member_certainty_std | member_certainty_range | prediction_dissimilarity_kl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 1000000 | max | 72 | 26.978664 | 35.123608 | 8.144943 | 0.665499 | 0.013445 | 0.052598 | 2.132449 |
| 64 | 1000000 | max | 36 | 16.547493 | 25.252651 | 8.705158 | 0.485100 | 0.023086 | 0.121536 | 3.561682 |
| 16 | 1000000 | min | 72 | 28.021307 | 34.731441 | 6.710135 | 0.658966 | 0.011257 | 0.039744 | 1.860980 |
| 64 | 1000000 | min | 36 | 17.351521 | 24.502899 | 7.151379 | 0.468377 | 0.026541 | 0.135208 | 3.081052 |
| 16 | 10000000 | max | 228 | 42.004078 | 63.011120 | 21.007042 | 0.857589 | 0.058578 | 0.189839 | 1.762257 |
| 16 | 10000000 | min | 228 | 44.725151 | 58.617992 | 13.892841 | 0.813459 | 0.058006 | 0.183929 | 1.686881 |

## Validation Loss

The validation-loss plot compares final held-out ensemble loss against `K`, with separate traces for scheduler and parameter budget. Use `validation_loss_vs_k.html` for the interactive version or `validation_loss_vs_k.png` for a static image.

## Training Curves

The training curves show the capped 256-update budget. Use `training_curves_all.html` for the interactive version, `training_curves_all.png` for the static overview, and `curve_k*_p*_*.png` for individual static plots.

## Ensembling Analysis

- Best final held-out ensemble loss: 16.547493 at K=64, params=1000000, scheduler=max.
- Certainty consistency: member-certainty ranges averaged by scheduler were {"max": 0.12132434050242107, "min": 0.11962702870368958}. Larger ranges indicate that some final sequence members are consistently more certain.
- Ensemble benefit: average individual CE minus ensemble CE averaged by parameter budget was {"1000000": 7.677903652191162, "10000000": 17.449941635131836}. Positive values mean ensembling improved held-out CE over the average individual member.
- Prediction dissimilarity: mean KL-to-ensemble by scheduler was {"max": 2.485462466875712, "min": 2.2096378405888877}. Higher values indicate more diverse member predictions.

## Additional Findings

The main engineering bottleneck is no longer validation memory. Training and evaluation both use exact chunked hidden-state computations, so the code never needs to materialize the full `[batch, K, seq_len, vocab]` logits tensor. The remaining cost is the repeated Latin-square routing depth and full-vocabulary projection over the active columns.

The run is resumable. Each completed cell writes a result JSON and curve CSV; rerunning the same command skips existing cells and regenerates the summary, plots, and writeup from all completed results.
