# Latin Square Ensemble Experiment

## What Was Run

This run scans the Latin-square ensemble over `K in {16, 64, 128}`, submodel budgets `{1M, 10M}`, and both min-entropy and max-entropy schedulers. The data is Wikitext-103 raw text tokenized with a 32,000-token BPE vocabulary and sequence length 128.

The architecture used here activates all Latin-square columns on each forward pass. For example, `K=128` evaluates all 128 ensemble paths instead of subsampling columns.

Training uses AdamW with weight decay 0.1 and gradient accumulation where needed to keep the effective batch size comparable while fitting memory. The run is deliberately capped at a comparable early-convergence budget rather than full Wikitext epochs: the training curves from the previous pilot showed that most loss reduction happened by about 256 optimizer updates. The exact microbatch sizes, accumulation factors, effective batch sizes, and token exposure are recorded below.

## Run Scope

| k | total_params | scheduler | max_train_batches | max_eval_batches | effective_train_batch_size | micro_train_batch_size | grad_accum_steps | effective_eval_batch_size | active_column_fraction | active_column_count | train_sequences_seen | train_tokens_seen | eval_sequences_seen | eval_tokens_seen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 1000000 | max | 128 | None | 12 | 1 | 12 | 1 | 1.00 | 16 | 3072 | 393216 | 1888 | 241664 |
| 64 | 1000000 | max | 128 | None | 12 | 1 | 12 | 1 | 1.00 | 64 | 3072 | 393216 | 1888 | 241664 |
| 128 | 1000000 | max | 128 | None | 12 | 1 | 12 | 1 | 1.00 | 128 | 3072 | 393216 | 1888 | 241664 |
| 16 | 1000000 | min | 128 | None | 12 | 1 | 12 | 1 | 1.00 | 16 | 3072 | 393216 | 1888 | 241664 |
| 64 | 1000000 | min | 128 | None | 12 | 1 | 12 | 1 | 1.00 | 64 | 3072 | 393216 | 1888 | 241664 |
| 128 | 1000000 | min | 128 | None | 12 | 1 | 12 | 1 | 1.00 | 128 | 3072 | 393216 | 1888 | 241664 |
| 16 | 10000000 | max | 128 | None | 12 | 1 | 12 | 1 | 1.00 | 16 | 3072 | 393216 | 1888 | 241664 |
| 64 | 10000000 | max | 128 | None | 12 | 1 | 12 | 1 | 1.00 | 64 | 3072 | 393216 | 1888 | 241664 |
| 128 | 10000000 | max | 128 | None | 12 | 1 | 12 | 1 | 1.00 | 128 | 3072 | 393216 | 1888 | 241664 |
| 16 | 10000000 | min | 128 | None | 12 | 1 | 12 | 1 | 1.00 | 16 | 3072 | 393216 | 1888 | 241664 |
| 64 | 10000000 | min | 128 | None | 12 | 1 | 12 | 1 | 1.00 | 64 | 3072 | 393216 | 1888 | 241664 |
| 128 | 10000000 | min | 128 | None | 12 | 1 | 12 | 1 | 1.00 | 128 | 3072 | 393216 | 1888 | 241664 |

## Results Table

| k | total_params | scheduler | d_model | ensemble_loss | avg_individual_loss | ensemble_gain | member_certainty_mean | member_certainty_std | member_certainty_range | prediction_dissimilarity_kl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 1000000 | max | 72 | 8.845925 | 24.364573 | 15.518649 | 0.256416 | 0.142414 | 0.428546 | 2.628227 |
| 64 | 1000000 | max | 36 | 8.415011 | 20.968193 | 12.553182 | 0.341803 | 0.291925 | 0.934946 | 3.589420 |
| 128 | 1000000 | max | 24 | 8.410655 | 18.570316 | 10.159661 | 0.318921 | 0.276602 | 0.952949 | 3.688416 |
| 16 | 1000000 | min | 72 | 9.255349 | 24.105746 | 14.850397 | 0.281411 | 0.173474 | 0.588636 | 2.523219 |
| 64 | 1000000 | min | 36 | 8.782350 | 20.415457 | 11.633107 | 0.337423 | 0.242346 | 0.858379 | 3.215899 |
| 128 | 1000000 | min | 24 | 8.739229 | 18.166058 | 9.426829 | 0.288020 | 0.222157 | 0.788009 | 3.213599 |
| 16 | 10000000 | max | 228 | 8.814678 | 36.616774 | 27.802097 | 0.305702 | 0.215366 | 0.593601 | 2.670201 |
| 64 | 10000000 | max | 114 | 8.245663 | 31.413189 | 23.167526 | 0.356261 | 0.294116 | 0.927854 | 3.783631 |
| 128 | 10000000 | max | 80 | 8.140687 | 28.839785 | 20.699099 | 0.407633 | 0.322574 | 0.930376 | 4.181162 |
| 16 | 10000000 | min | 228 | 9.077996 | 34.288267 | 25.210272 | 0.312703 | 0.240630 | 0.830918 | 2.537532 |
| 64 | 10000000 | min | 114 | 8.398748 | 30.360224 | 21.961476 | 0.351048 | 0.285701 | 0.911720 | 3.572804 |
| 128 | 10000000 | min | 80 | 8.315472 | 27.601649 | 19.286177 | 0.385716 | 0.305576 | 0.923748 | 3.861725 |

## Validation Loss

The validation-loss plot compares final held-out ensemble loss against `K`, with separate traces for scheduler and parameter budget. Use `validation_loss_vs_k.html` for the interactive version or `validation_loss_vs_k.png` for a static image.

## Training Curves

The training curves show the capped 256-update budget. Use `training_curves_all.html` for the interactive version, `training_curves_all.png` for the static overview, and `curve_k*_p*_*.png` for individual static plots.

## Ensembling Analysis

- Best final held-out ensemble loss: 8.140687 at K=128, params=10000000, scheduler=max.
- Certainty consistency: member-certainty ranges averaged by scheduler were {"max": 0.7947119381278753, "min": 0.8169016918788353}. Larger ranges indicate that some final sequence members are consistently more certain.
- Ensemble benefit: average individual CE minus ensemble CE averaged by parameter budget was {"1000000": 12.356970849557449, "10000000": 23.021107681642818}. Positive values mean ensembling improved held-out CE over the average individual member.
- Prediction dissimilarity: mean KL-to-ensemble by scheduler was {"max": 3.423509457269631, "min": 3.1541295133236438}. Higher values indicate more diverse member predictions.

## Additional Findings

The main engineering bottleneck is no longer validation memory. Training and evaluation both use exact chunked hidden-state computations, so the code never needs to materialize the full `[batch, K, seq_len, vocab]` logits tensor. The remaining cost is the repeated Latin-square routing depth and full-vocabulary projection over the active columns.

The run is resumable. Each completed cell writes a result JSON and curve CSV; rerunning the same command skips existing cells and regenerates the summary, plots, and writeup from all completed results.
