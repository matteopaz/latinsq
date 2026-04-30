# Latin Square Ensemble Experiment

## What Was Run

This run scans the Latin-square ensemble over `K in {16, 64, 128}`, submodel budgets `{1M, 10M}`, and both min-entropy and max-entropy schedulers. The data is Wikitext-103 raw text tokenized with a 32,000-token BPE vocabulary and sequence length 128.

The architecture used here activates all Latin-square columns on each forward pass. For example, `K=128` evaluates all 128 ensemble paths instead of subsampling columns.

Training uses AdamW with weight decay 0.1 and gradient accumulation where needed to keep the effective batch size comparable while fitting memory. The run is deliberately capped at a comparable early-convergence budget rather than full Wikitext epochs: the training curves from the previous pilot showed that most loss reduction happened by about 256 optimizer updates. The exact microbatch sizes, accumulation factors, effective batch sizes, and token exposure are recorded below.

## Run Scope

| k | total_params | scheduler | max_train_batches | max_eval_batches | effective_train_batch_size | micro_train_batch_size | grad_accum_steps | effective_eval_batch_size | active_column_fraction | active_column_count | train_sequences_seen | train_tokens_seen | eval_sequences_seen | eval_tokens_seen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 1000000 | max | 1 | 1 | 12 | 1 | 12 | 1 | 1.00 | 16 | 24 | 3072 | 1 | 128 |
| 64 | 1000000 | max | 1 | 1 | 12 | 1 | 12 | 1 | 1.00 | 64 | 24 | 3072 | 1 | 128 |
| 128 | 1000000 | max | 1 | 1 | 12 | 1 | 12 | 1 | 1.00 | 128 | 24 | 3072 | 1 | 128 |
| 16 | 1000000 | min | 1 | 1 | 12 | 1 | 12 | 1 | 1.00 | 16 | 24 | 3072 | 1 | 128 |
| 64 | 1000000 | min | 1 | 1 | 12 | 1 | 12 | 1 | 1.00 | 64 | 24 | 3072 | 1 | 128 |
| 128 | 1000000 | min | 1 | 1 | 12 | 1 | 12 | 1 | 1.00 | 128 | 24 | 3072 | 1 | 128 |
| 16 | 10000000 | max | 1 | 1 | 12 | 1 | 12 | 1 | 1.00 | 16 | 24 | 3072 | 1 | 128 |
| 64 | 10000000 | max | 1 | 1 | 12 | 1 | 12 | 1 | 1.00 | 64 | 24 | 3072 | 1 | 128 |
| 128 | 10000000 | max | 1 | 1 | 12 | 1 | 12 | 1 | 1.00 | 128 | 24 | 3072 | 1 | 128 |
| 16 | 10000000 | min | 1 | 1 | 12 | 1 | 12 | 1 | 1.00 | 16 | 24 | 3072 | 1 | 128 |
| 64 | 10000000 | min | 1 | 1 | 12 | 1 | 12 | 1 | 1.00 | 64 | 24 | 3072 | 1 | 128 |
| 128 | 10000000 | min | 1 | 1 | 12 | 1 | 12 | 1 | 1.00 | 128 | 24 | 3072 | 1 | 128 |

## Results Table

| k | total_params | scheduler | d_model | ensemble_loss | avg_individual_loss | ensemble_gain | member_certainty_mean | member_certainty_std | member_certainty_range | prediction_dissimilarity_kl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 1000000 | max | 72 | 27.101128 | 35.586418 | 8.485291 | 0.657870 | 0.020867 | 0.079596 | 2.218048 |
| 64 | 1000000 | max | 36 | 16.698317 | 25.284386 | 8.586069 | 0.487245 | 0.045097 | 0.238192 | 3.546600 |
| 128 | 1000000 | max | 24 | 14.154517 | 21.090626 | 6.936109 | 0.370686 | 0.065119 | 0.411687 | 3.846263 |
| 16 | 1000000 | min | 72 | 27.829027 | 34.804482 | 6.975454 | 0.673018 | 0.022731 | 0.079288 | 1.700640 |
| 64 | 1000000 | min | 36 | 17.221592 | 24.784275 | 7.562683 | 0.510023 | 0.056265 | 0.281845 | 3.145881 |
| 128 | 1000000 | min | 24 | 13.792578 | 21.184948 | 7.392370 | 0.375059 | 0.057459 | 0.302633 | 3.435024 |
| 16 | 10000000 | max | 228 | 39.863281 | 57.937759 | 18.074478 | 0.817241 | 0.036703 | 0.125812 | 1.860236 |
| 64 | 10000000 | max | 114 | 21.899147 | 44.930225 | 23.031078 | 0.774411 | 0.105821 | 0.349064 | 3.535602 |
| 128 | 10000000 | max | 80 | 15.467407 | 35.089745 | 19.622337 | 0.677942 | 0.094473 | 0.448963 | 4.291142 |
| 16 | 10000000 | min | 228 | 40.501633 | 56.016487 | 15.514854 | 0.807853 | 0.032472 | 0.133985 | 1.725011 |
| 64 | 10000000 | min | 114 | 22.960358 | 42.732956 | 19.772598 | 0.771892 | 0.100857 | 0.371694 | 3.030389 |
| 128 | 10000000 | min | 80 | 19.381863 | 34.679100 | 15.297237 | 0.666927 | 0.079150 | 0.446367 | 3.787277 |

## Validation Loss

The validation-loss plot compares final held-out ensemble loss against `K`, with separate traces for scheduler and parameter budget. Use `validation_loss_vs_k.html` for the interactive version or `validation_loss_vs_k.png` for a static image.

## Training Curves

The training curves show the capped 256-update budget. Use `training_curves_all.html` for the interactive version, `training_curves_all.png` for the static overview, and `curve_k*_p*_*.png` for individual static plots.

## Ensembling Analysis

- Best final held-out ensemble loss: 13.792578 at K=128, params=1000000, scheduler=min.
- Certainty consistency: member-certainty ranges averaged by scheduler were {"max": 0.2755521858731906, "min": 0.2693023011088371}. Larger ranges indicate that some final sequence members are consistently more certain.
- Ensemble benefit: average individual CE minus ensemble CE averaged by parameter budget was {"1000000": 7.656329313913981, "10000000": 18.55209732055664}. Positive values mean ensembling improved held-out CE over the average individual member.
- Prediction dissimilarity: mean KL-to-ensemble by scheduler was {"max": 3.2163150310516357, "min": 2.804037014643351}. Higher values indicate more diverse member predictions.

## Additional Findings

The main engineering bottleneck is no longer validation memory. Training and evaluation both use exact chunked hidden-state computations, so the code never needs to materialize the full `[batch, K, seq_len, vocab]` logits tensor. The remaining cost is the repeated Latin-square routing depth and full-vocabulary projection over the active columns.

The run is resumable. Each completed cell writes a result JSON and curve CSV; rerunning the same command skips existing cells and regenerates the summary, plots, and writeup from all completed results.
