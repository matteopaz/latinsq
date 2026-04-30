# Latin Square Ensemble Experiment

## What Was Run

This run scans the Latin-square ensemble over `K in {16, 64, 128}`, submodel budgets `{1M, 10M}`, and both min-entropy and max-entropy schedulers. The data is Wikitext-103 raw text tokenized with a 32,000-token BPE vocabulary and sequence length 128.

The architecture used here activates only a quarter of the Latin-square columns on each forward pass. For example, `K=128` evaluates 32 ensemble paths per batch instead of all 128. Inactive columns are sampled on later forwards, so the run keeps the global Latin-square structure while reducing per-step compute and memory.

Training uses AdamW with weight decay 0.1. The run is deliberately capped at a comparable early-convergence budget rather than full Wikitext epochs: the training curves from the previous pilot showed that most loss reduction happened by about 256 optimizer updates. The exact batch caps, effective batch sizes, and token exposure are recorded below.

## Run Scope

| k | total_params | scheduler | max_train_batches | max_eval_batches | effective_train_batch_size | effective_eval_batch_size | active_column_fraction | active_column_count | train_sequences_seen | train_tokens_seen | eval_sequences_seen | eval_tokens_seen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 1000000 | max | 128 | None | 12 | 12 | 0.25 | 4 | 3072 | 393216 | 1896 | 242688 |
| 64 | 1000000 | max | 128 | None | 12 | 12 | 0.25 | 16 | 3072 | 393216 | 1896 | 242688 |
| 128 | 1000000 | max | 128 | None | 12 | 12 | 0.25 | 32 | 3072 | 393216 | 1896 | 242688 |
| 16 | 1000000 | min | 128 | None | 12 | 12 | 0.25 | 4 | 3072 | 393216 | 1896 | 242688 |
| 64 | 1000000 | min | 128 | None | 12 | 12 | 0.25 | 16 | 3072 | 393216 | 1896 | 242688 |
| 128 | 1000000 | min | 128 | None | 12 | 12 | 0.25 | 32 | 3072 | 393216 | 1896 | 242688 |
| 16 | 10000000 | max | 128 | None | 12 | 12 | 0.25 | 4 | 3072 | 393216 | 1896 | 242688 |
| 64 | 10000000 | max | 128 | None | 12 | 12 | 0.25 | 16 | 3072 | 393216 | 1896 | 242688 |
| 128 | 10000000 | max | 128 | None | 12 | 12 | 0.25 | 32 | 3072 | 393216 | 1896 | 242688 |
| 16 | 10000000 | min | 128 | None | 12 | 12 | 0.25 | 4 | 3072 | 393216 | 1896 | 242688 |
| 64 | 10000000 | min | 128 | None | 12 | 12 | 0.25 | 16 | 3072 | 393216 | 1896 | 242688 |
| 128 | 10000000 | min | 128 | None | 12 | 12 | 0.25 | 32 | 3072 | 393216 | 1896 | 242688 |

## Results Table

| k | total_params | scheduler | d_model | ensemble_loss | avg_individual_loss | ensemble_gain | member_certainty_mean | member_certainty_std | member_certainty_range | prediction_dissimilarity_kl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 1000000 | max | 72 | 13.046841 | 20.636220 | 7.589379 | 0.105697 | 0.028035 | 0.110795 | 0.853087 |
| 64 | 1000000 | max | 36 | 10.229288 | 19.339809 | 9.110521 | 0.176573 | 0.096889 | 0.540741 | 1.977490 |
| 128 | 1000000 | max | 24 | 9.540576 | 17.754146 | 8.213570 | 0.209435 | 0.132852 | 0.544704 | 2.516618 |
| 16 | 1000000 | min | 72 | 12.994503 | 20.960924 | 7.966422 | 0.110926 | 0.033014 | 0.120125 | 0.915036 |
| 64 | 1000000 | min | 36 | 10.282847 | 19.421798 | 9.138951 | 0.181354 | 0.093911 | 0.457163 | 2.011086 |
| 128 | 1000000 | min | 24 | 9.467301 | 17.517693 | 8.050392 | 0.222183 | 0.121183 | 0.628605 | 2.329833 |
| 16 | 10000000 | max | 228 | 15.165534 | 27.065380 | 11.899846 | 0.182484 | 0.089500 | 0.529281 | 0.796141 |
| 64 | 10000000 | max | 114 | 10.377061 | 27.338874 | 16.961813 | 0.219502 | 0.140733 | 0.728371 | 1.942385 |
| 128 | 10000000 | max | 80 | 9.369934 | 25.286706 | 15.916772 | 0.218417 | 0.140092 | 0.790809 | 2.471614 |
| 16 | 10000000 | min | 228 | 13.647183 | 27.133616 | 13.486432 | 0.163606 | 0.042519 | 0.228111 | 0.664335 |
| 64 | 10000000 | min | 114 | 10.126769 | 26.432828 | 16.306058 | 0.163083 | 0.088847 | 0.515042 | 1.750924 |
| 128 | 10000000 | min | 80 | 9.483265 | 25.126277 | 15.643012 | 0.204946 | 0.116108 | 0.539831 | 2.299443 |

## Validation Loss

The validation-loss plot compares final held-out ensemble loss against `K`, with separate traces for scheduler and parameter budget. Use `validation_loss_vs_k.html` for the interactive version or `validation_loss_vs_k.png` for a static image.

## Training Curves

The training curves show the capped 256-update budget. Use `training_curves_all.html` for the interactive version, `training_curves_all.png` for the static overview, and `curve_k*_p*_*.png` for individual static plots.

## Ensembling Analysis

- Best final held-out ensemble loss: 9.369934 at K=128, params=10000000, scheduler=max.
- Certainty consistency: member-certainty ranges averaged by scheduler were {"max": 0.5407833661884069, "min": 0.41481270268559456}. Larger ranges indicate that some final sequence members are consistently more certain.
- Ensemble benefit: average individual CE minus ensemble CE averaged by parameter budget was {"1000000": 8.344872341880317, "10000000": 15.035655711773579}. Positive values mean ensembling improved held-out CE over the average individual member.
- Prediction dissimilarity: mean KL-to-ensemble by scheduler was {"max": 1.759555880797689, "min": 1.6617761191829474}. Higher values indicate more diverse member predictions.

## Additional Findings

The main engineering bottleneck is no longer validation memory. Training and evaluation both use exact chunked hidden-state computations, so the code never needs to materialize the full `[batch, K, seq_len, vocab]` logits tensor. The remaining cost is the repeated Latin-square routing depth and full-vocabulary projection over the active columns.

The run is resumable. Each completed cell writes a result JSON and curve CSV; rerunning the same command skips existing cells and regenerates the summary, plots, and writeup from all completed results.
