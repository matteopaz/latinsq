# Latin Square Ensemble Experiment

## What Was Run

This run scans the Latin-square ensemble over `K in {16, 64, 128}`, submodel budgets `{1M, 10M}`, and both min-entropy and max-entropy schedulers. The data is Wikitext-103 raw text tokenized with a 32,000-token BPE vocabulary and sequence length 128.

The architecture used here activates 0.12 of the Latin-square columns on each forward pass. For example, `K=128` evaluates 16 ensemble paths per batch instead of all 128. Inactive columns are sampled on later forwards, so the run keeps the global Latin-square structure while reducing per-step compute and memory.

Training uses AdamW with weight decay 0.1. The run is deliberately capped at a comparable early-convergence budget rather than full Wikitext epochs: the training curves from the previous pilot showed that most loss reduction happened by about 256 optimizer updates. The exact batch caps, effective batch sizes, and token exposure are recorded below.

## Run Scope

| k | total_params | scheduler | max_train_batches | max_eval_batches | effective_train_batch_size | micro_train_batch_size | grad_accum_steps | effective_eval_batch_size | active_column_fraction | active_row_fraction | active_column_count | active_row_count | train_sequences_seen | train_tokens_seen | eval_sequences_seen | eval_tokens_seen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 10000000 | max | 3267 | None | 12 | 12 | 1 | 12 | 0.12 | 0.50 | 16 | 64 | 39204 | 5018112 | 1888 | 241664 |

## Results Table

| k | total_params | scheduler | d_model | ensemble_loss | avg_individual_loss | ensemble_gain | member_certainty_mean | member_certainty_std | member_certainty_range | prediction_dissimilarity_kl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 10000000 | max | 80 | 7.321269 | 15.758792 | 8.437524 | 0.151620 | 0.062999 | 0.220911 | 1.246632 |

## Validation Loss

The validation-loss plot compares final held-out ensemble loss against `K`, with separate traces for scheduler and parameter budget. Use `validation_loss_vs_k.html` for the interactive version or `validation_loss_vs_k.png` for a static image.

## Training Curves

The training curves show the capped 256-update budget. Use `training_curves_all.html` for the interactive version, `training_curves_all.png` for the static overview, and `curve_k*_p*_*.png` for individual static plots.

## Ensembling Analysis

- Best final held-out ensemble loss: 7.321269 at K=128, params=10000000, scheduler=max.
- Certainty consistency: member-certainty ranges averaged by scheduler were {"max": 0.22091111168265343}. Larger ranges indicate that some final sequence members are consistently more certain.
- Ensemble benefit: average individual CE minus ensemble CE averaged by parameter budget was {"10000000": 8.437523645690725}. Positive values mean ensembling improved held-out CE over the average individual member.
- Prediction dissimilarity: mean KL-to-ensemble by scheduler was {"max": 1.2466318890263763}. Higher values indicate more diverse member predictions.

## Additional Findings

The main engineering bottleneck is no longer validation memory. Training and evaluation both use exact chunked hidden-state computations, so the code never needs to materialize the full `[batch, K, seq_len, vocab]` logits tensor. The remaining cost is the repeated Latin-square routing depth and full-vocabulary projection over the active columns.

The run is resumable. Each completed cell writes a result JSON and curve CSV; rerunning the same command skips existing cells and regenerates the summary, plots, and writeup from all completed results.
