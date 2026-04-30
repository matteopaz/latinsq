# GPT-Style Parameter-Matched Traditional LM Comparison

This table matches each all-columns Latin-square row to a GPT-style conventional Transformer trained with the same 393,216-token exposure. The baseline uses GPT-style initialization, residual projection scaling, and output logits scaled by `1/sqrt(d_model)`.

| latin_budget_label | k | scheduler | latin_actual_params | traditional_actual_params | traditional_d_model | traditional_layers | traditional_heads | param_ratio_traditional_over_latin | latin_ensemble_loss | traditional_validation_loss | loss_delta_traditional_minus_latin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000000 | 16 | max | 3323664 | 3320382 | 78 | 11 | 1 | 0.999013 | 8.845925 | 9.606232 | 0.760307 |
| 1000000 | 16 | min | 3323664 | 3320382 | 78 | 11 | 1 | 0.999013 | 9.255349 | 9.606232 | 0.350882 |
| 1000000 | 64 | max | 2181960 | 2174315 | 55 | 11 | 1 | 0.996496 | 8.415011 | 9.720547 | 1.305536 |
| 1000000 | 64 | min | 2181960 | 2174315 | 55 | 11 | 1 | 0.996496 | 8.782350 | 9.720547 | 0.938197 |
| 1000000 | 128 | max | 1695792 | 1694700 | 45 | 10 | 1 | 0.999356 | 8.410655 | 9.775051 | 1.364396 |
| 1000000 | 128 | min | 1695792 | 1694700 | 45 | 10 | 1 | 0.999356 | 8.739229 | 9.775051 | 1.035822 |
| 10000000 | 16 | max | 17353992 | 17314180 | 260 | 11 | 4 | 0.997706 | 8.814678 | 9.078279 | 0.263602 |
| 10000000 | 16 | min | 17353992 | 17314180 | 260 | 11 | 4 | 0.997706 | 9.077996 | 9.078279 | 0.000284 |
| 10000000 | 64 | max | 13738596 | 13692240 | 216 | 12 | 4 | 0.996626 | 8.245663 | 9.174719 | 0.929056 |
| 10000000 | 64 | min | 13738596 | 13692240 | 216 | 12 | 4 | 0.996626 | 8.398748 | 9.174719 | 0.775971 |
| 10000000 | 128 | max | 12533920 | 12423632 | 208 | 11 | 4 | 0.991201 | 8.140687 | 9.192089 | 1.051402 |
| 10000000 | 128 | min | 12533920 | 12423632 | 208 | 11 | 4 | 0.991201 | 8.315472 | 9.192089 | 0.876617 |
