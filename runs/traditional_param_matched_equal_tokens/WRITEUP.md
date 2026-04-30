# Traditional LM Baseline

## What Was Run

This trains a single conventional causal Transformer language model at two target trainable-parameter budgets: `1M` and `10M`. It uses the same Wikitext-103 BPE cache, vocabulary size 32,000, sequence length 128, AdamW settings, 256 optimizer updates, and 393,216 training tokens per run as the all-columns Latin-square run.

The model uses tied input/output token embeddings, learned positional embeddings, two causal self-attention blocks, two attention heads, and a width chosen as the largest multiple of the head count that stays under the requested trainable-parameter budget.

## Results

| target_params | actual_trainable_params | n_layers | n_heads | d_model | validation_loss | certainty_mean | train_tokens_seen | eval_tokens_seen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1695792 | 1667800 | 2 | 2 | 50 | 14.364551 | 0.059020 | 393216 | 241664 |
| 2181960 | 2156288 | 2 | 2 | 64 | 14.577940 | 0.056363 | 393216 | 241664 |
| 3323664 | 3308160 | 2 | 2 | 96 | 14.447372 | 0.071180 | 393216 | 241664 |
| 12533920 | 12463288 | 2 | 2 | 314 | 14.785290 | 0.098245 | 393216 | 241664 |
| 13738596 | 13707440 | 2 | 2 | 340 | 14.539259 | 0.130258 | 393216 | 241664 |
| 17353992 | 17322128 | 2 | 2 | 412 | 14.745344 | 0.124596 | 393216 | 241664 |

## Plots

Interactive training curves are in `training_curves_traditional.html`. Interactive validation-loss comparison by parameter budget is in `validation_loss_vs_params_traditional.html`.

## Comparison Note

These baselines are parameter-capped including tied embeddings and all Transformer weights. The Latin-square scan's `total_params` field follows the experiment instruction and excludes embeddings/head from the submodel budget, so compare validation losses as same-exposure baselines rather than as perfectly matched total trainable-parameter counts.
