# Latin Square Ensemble Experiment

## What Was Run

This run scans the Latin-square ensemble over `K in {16, 64, 128}`, submodel budgets `{1M, 10M}`, and both min-entropy and max-entropy schedulers. The data is Wikitext-103 raw text tokenized with a 32,000-token BPE vocabulary and sequence length 128.

The architecture used here activates only a quarter of the Latin-square columns on each forward pass. For example, `K=128` evaluates 32 ensemble paths per batch instead of all 128. Inactive columns are sampled on later forwards, so the run keeps the global Latin-square structure while reducing per-step compute and memory.

Training uses AdamW with weight decay 0.1. The run is deliberately capped at a comparable early-convergence budget rather than full Wikitext epochs: the training curves from the previous pilot showed that most loss reduction happened by about 256 optimizer updates. The exact batch caps, effective batch sizes, and token exposure are recorded below.

## Run Scope

| k | total_params | scheduler | max_train_batches | max_eval_batches | effective_train_batch_size | effective_eval_batch_size | active_column_fraction | active_column_count | train_sequences_seen | train_tokens_seen | eval_sequences_seen | eval_tokens_seen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 1000000 | max | 128 | None | 96 | 16 | 0.25 | 4 | 24576 | 3145728 | 1888 | 241664 |
| 64 | 1000000 | max | 128 | None | 24 | 16 | 0.25 | 16 | 6144 | 786432 | 1888 | 241664 |
| 128 | 1000000 | max | 128 | None | 12 | 12 | 0.25 | 32 | 3072 | 393216 | 1896 | 242688 |
| 16 | 1000000 | min | 128 | None | 96 | 16 | 0.25 | 4 | 24576 | 3145728 | 1888 | 241664 |
| 64 | 1000000 | min | 128 | None | 24 | 16 | 0.25 | 16 | 6144 | 786432 | 1888 | 241664 |
| 128 | 1000000 | min | 128 | None | 12 | 12 | 0.25 | 32 | 3072 | 393216 | 1896 | 242688 |
| 16 | 10000000 | max | 128 | None | 96 | 16 | 0.25 | 4 | 24576 | 3145728 | 1888 | 241664 |
| 64 | 10000000 | max | 128 | None | 24 | 16 | 0.25 | 16 | 6144 | 786432 | 1888 | 241664 |
| 128 | 10000000 | max | 128 | None | 12 | 12 | 0.25 | 32 | 3072 | 393216 | 1896 | 242688 |
| 16 | 10000000 | min | 128 | None | 96 | 16 | 0.25 | 4 | 24576 | 3145728 | 1888 | 241664 |
| 64 | 10000000 | min | 128 | None | 24 | 16 | 0.25 | 16 | 6144 | 786432 | 1888 | 241664 |
| 128 | 10000000 | min | 128 | None | 12 | 12 | 0.25 | 32 | 3072 | 393216 | 1896 | 242688 |

## Results Table

| k | total_params | scheduler | d_model | ensemble_loss | avg_individual_loss | ensemble_gain | member_certainty_mean | member_certainty_std | member_certainty_range | prediction_dissimilarity_kl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 1000000 | max | 72 | 12.411936 | 20.177507 | 7.765571 | 0.113902 | 0.027827 | 0.118161 | 0.878680 |
| 64 | 1000000 | max | 36 | 10.027356 | 19.489829 | 9.462473 | 0.163505 | 0.082050 | 0.378524 | 2.012609 |
| 128 | 1000000 | max | 24 | 9.633757 | 17.861037 | 8.227280 | 0.200434 | 0.143380 | 0.676198 | 2.535186 |
| 16 | 1000000 | min | 72 | 12.874239 | 21.218633 | 8.344394 | 0.121463 | 0.028065 | 0.107119 | 0.832715 |
| 64 | 1000000 | min | 36 | 9.978204 | 19.109345 | 9.131141 | 0.197126 | 0.089028 | 0.366040 | 1.925780 |
| 128 | 1000000 | min | 24 | 9.600918 | 17.665611 | 8.064693 | 0.200327 | 0.141168 | 0.544498 | 2.319251 |
| 16 | 10000000 | max | 228 | 12.637574 | 26.497373 | 13.859799 | 0.218549 | 0.068647 | 0.390183 | 0.753141 |
| 64 | 10000000 | max | 114 | 10.400436 | 27.421509 | 17.021073 | 0.299471 | 0.218294 | 0.914485 | 2.008180 |
| 128 | 10000000 | max | 80 | 9.460799 | 25.703676 | 16.242878 | 0.270730 | 0.194743 | 0.950140 | 2.508083 |
| 16 | 10000000 | min | 228 | 12.704550 | 25.626030 | 12.921480 | 0.156109 | 0.064257 | 0.272147 | 0.728852 |
| 64 | 10000000 | min | 114 | 10.300577 | 27.747601 | 17.447024 | 0.183384 | 0.079372 | 0.422898 | 1.993221 |
| 128 | 10000000 | min | 80 | 9.493030 | 25.021057 | 15.528028 | 0.177744 | 0.081846 | 0.542095 | 2.361584 |

## Validation Loss

The validation-loss plot compares final held-out ensemble loss against `K`, with separate traces for scheduler and parameter budget. Use `validation_loss_vs_k.html` for the interactive version or `validation_loss_vs_k.png` for a static image.

## Training Curves

The training curves show the capped 256-update budget. Use `training_curves_all.html` for the interactive version, `training_curves_all.png` for the static overview, and `curve_k*_p*_*.png` for individual static plots.

## Ensembling Analysis

- Best final held-out ensemble loss: 9.460799 at K=128, params=10000000, scheduler=max.
- Certainty consistency: member-certainty ranges averaged by scheduler were {"max": 0.5712818118433157, "min": 0.3757996838539839}. Larger ranges indicate that some final sequence members are consistently more certain.
- Ensemble benefit: average individual CE minus ensemble CE averaged by parameter budget was {"1000000": 8.49925853480923, "10000000": 15.503380123538912}. Positive values mean ensembling improved held-out CE over the average individual member.
- Prediction dissimilarity: mean KL-to-ensemble by scheduler was {"max": 1.7826464400543929, "min": 1.693567147183406}. Higher values indicate more diverse member predictions.

## Additional Findings

The main engineering bottleneck is no longer validation memory. Training and evaluation both use exact chunked hidden-state computations, so the code never needs to materialize the full `[batch, K, seq_len, vocab]` logits tensor. The remaining cost is the repeated Latin-square routing depth and full-vocabulary projection over the active columns.

The run is resumable. Each completed cell writes a result JSON and curve CSV; rerunning the same command skips existing cells and regenerates the summary, plots, and writeup from all completed results.
