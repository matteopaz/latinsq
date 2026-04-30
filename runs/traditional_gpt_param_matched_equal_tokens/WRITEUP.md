# Traditional LM Baseline

## What Was Run

This trains a single conventional GPT-style causal Transformer language model at the requested trainable-parameter targets. It uses the same Wikitext-103 BPE cache, vocabulary size 32,000, sequence length 128, AdamW settings, 256 optimizer updates, and 393,216 training tokens per run as the all-columns Latin-square run.

The model uses tied input/output token embeddings, learned positional embeddings, pre-LN causal self-attention blocks, GPT-style normal initialization, residual projection scaling, and logits scaled by `1/sqrt(d_model)`. Unless fixed depth/head settings are supplied, the runner searches over conventional depth, width, and head-count choices and selects the largest stable GPT-style architecture under the target parameter budget.

## Results

| target_params | actual_trainable_params | n_layers | n_heads | d_model | validation_loss | certainty_mean | train_tokens_seen | eval_tokens_seen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1695792 | 1694700 | 10 | 1 | 45 | 9.775051 | 0.000089 | 393216 | 241664 |
| 2181960 | 2174315 | 11 | 1 | 55 | 9.720547 | 0.000098 | 393216 | 241664 |
| 3323664 | 3320382 | 11 | 1 | 78 | 9.606232 | 0.000121 | 393216 | 241664 |
| 12533920 | 12423632 | 11 | 4 | 208 | 9.192089 | 0.000263 | 393216 | 241664 |
| 13738596 | 13692240 | 12 | 4 | 216 | 9.174719 | 0.000261 | 393216 | 241664 |
| 17353992 | 17314180 | 11 | 4 | 260 | 9.078279 | 0.000337 | 393216 | 241664 |

## Plots

Interactive training curves are in `training_curves_traditional.html`. Interactive validation-loss comparison by parameter budget is in `validation_loss_vs_params_traditional.html`.

## Comparison Note

These baselines are parameter-capped including tied embeddings and all Transformer weights. For parameter-matched reruns, the target parameter counts should be the Latin-square rows' actual trainable-parameter counts, not the nominal submodel-budget labels.
