# Traditional LM Baseline

## What Was Run

This trains a single conventional causal Transformer language model at two target trainable-parameter budgets: `1M` and `10M`. It uses the same Wikitext-103 BPE cache, vocabulary size 32,000, sequence length 128, AdamW settings, 256 optimizer updates, and 393,216 training tokens per run as the all-columns Latin-square run.

The model uses tied input/output token embeddings, learned positional embeddings, two causal self-attention blocks, two attention heads, and a width chosen as the largest multiple of the head count that stays under the requested trainable-parameter budget.

## Results

| target_params | actual_trainable_params | n_layers | n_heads | d_model | validation_loss | certainty_mean | train_tokens_seen | eval_tokens_seen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000000 | 986280 | 2 | 2 | 30 | 14.003551 | 0.048445 | 393216 | 241664 |
| 10000000 | 9982960 | 2 | 2 | 260 | 14.449874 | 0.104407 | 393216 | 241664 |

## Plots

Interactive training curves are in `training_curves_traditional.html`. Interactive validation-loss comparison by parameter budget is in `validation_loss_vs_params_traditional.html`.

## Comparison Note

These baselines are parameter-capped including tied embeddings and all Transformer weights. The Latin-square scan's `total_params` field follows the experiment instruction and excludes embeddings/head from the submodel budget, so compare validation losses as same-exposure baselines rather than as perfectly matched total trainable-parameter counts.
