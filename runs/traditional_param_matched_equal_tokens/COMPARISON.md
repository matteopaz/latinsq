# Parameter-Matched Traditional LM Comparison

This table matches each all-columns Latin-square row to a conventional two-layer tied-embedding Transformer trained with the same 393,216-token exposure. The traditional width is chosen as the largest valid width not exceeding the Latin row's actual trainable-parameter count, so it is usually slightly under the Latin count.

| latin_budget_label | k | scheduler | latin_actual_params | traditional_actual_params | param_ratio_traditional_over_latin | latin_ensemble_loss | traditional_validation_loss | loss_delta_traditional_minus_latin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000000 | 16 | max | 3323664 | 3308160 | 0.995335 | 8.845925 | 14.447372 | 5.601447 |
| 1000000 | 16 | min | 3323664 | 3308160 | 0.995335 | 9.255349 | 14.447372 | 5.192022 |
| 1000000 | 64 | max | 2181960 | 2156288 | 0.988234 | 8.415011 | 14.577940 | 6.162929 |
| 1000000 | 64 | min | 2181960 | 2156288 | 0.988234 | 8.782350 | 14.577940 | 5.795590 |
| 1000000 | 128 | max | 1695792 | 1667800 | 0.983493 | 8.410655 | 14.364551 | 5.953896 |
| 1000000 | 128 | min | 1695792 | 1667800 | 0.983493 | 8.739229 | 14.364551 | 5.625322 |
| 10000000 | 16 | max | 17353992 | 17322128 | 0.998164 | 8.814678 | 14.745344 | 5.930666 |
| 10000000 | 16 | min | 17353992 | 17322128 | 0.998164 | 9.077996 | 14.745344 | 5.667348 |
| 10000000 | 64 | max | 13738596 | 13707440 | 0.997732 | 8.245663 | 14.539259 | 6.293596 |
| 10000000 | 64 | min | 13738596 | 13707440 | 0.997732 | 8.398748 | 14.539259 | 6.140511 |
| 10000000 | 128 | max | 12533920 | 12463288 | 0.994365 | 8.140687 | 14.785290 | 6.644604 |
| 10000000 | 128 | min | 12533920 | 12463288 | 0.994365 | 8.315472 | 14.785290 | 6.469819 |
