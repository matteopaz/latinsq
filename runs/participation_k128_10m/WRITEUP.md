# Participation Analysis

This diagnostic retrains and evaluates the `k=128`, `10M` Latin-square model with all columns active, once with the max-entropy scheduler and once with the min-entropy scheduler.

For each scheduler, it evaluates sampled column subsets at sizes `1, 8, 16, 24, ..., 128`. The plotted point is the average validation ensemble loss across sampled subsets of that size. The vertical range bar spans the best to worst subset loss at the same size, so it measures how sensitive the ensemble is to which columns are chosen.

## Scope

- Training exposure per scheduler: `2 * 128` optimizer updates, effective batch `12`, sequence length `128`.
- Evaluation exposure per subset: `32` validation batches with batch size 1.
- Random column subsets per size: `8`, except the full 128-column point has one subset.
- The ensemble loss is computed by averaging member probabilities post-softmax, not by averaging logits.

## Files

- `participation_loss_vs_columns.html`: interactive plot with min/max range bars.
- `participation_loss_vs_columns.png`: static plot.
- `participation_loss_summary.csv`: one row per scheduler and subset size.
- `participation_loss_samples.csv`: one row per sampled column subset.
- `checkpoint_k128_p10000000_{max,min}.pt`: saved model checkpoints for reruns.

## Summary Table

| scheduler | columns | subsets | mean_loss | min_loss | max_loss | range |
| --- | --- | --- | --- | --- | --- | --- |
| max | 1 | 8 | 28.4104 | 26.7281 | 31.1547 | 4.4266 |
| max | 8 | 8 | 16.4227 | 15.3870 | 18.3548 | 2.9678 |
| max | 16 | 8 | 13.7531 | 12.9224 | 14.3632 | 1.4408 |
| max | 24 | 8 | 12.4835 | 11.6911 | 12.8614 | 1.1703 |
| max | 32 | 8 | 11.5028 | 11.0188 | 11.9971 | 0.9783 |
| max | 40 | 8 | 10.8444 | 10.4161 | 11.4019 | 0.9857 |
| max | 48 | 8 | 10.3839 | 10.0119 | 11.1187 | 1.1067 |
| max | 56 | 8 | 10.0269 | 9.5403 | 10.4236 | 0.8833 |
| max | 64 | 8 | 9.5251 | 9.2282 | 9.6477 | 0.4195 |
| max | 72 | 8 | 9.2433 | 9.1313 | 9.4181 | 0.2867 |
| max | 80 | 8 | 9.1543 | 8.9515 | 9.3112 | 0.3597 |
| max | 88 | 8 | 8.9249 | 8.8273 | 9.0244 | 0.1971 |
| max | 96 | 8 | 8.7116 | 8.6449 | 8.8045 | 0.1596 |
| max | 104 | 8 | 8.6332 | 8.5657 | 8.7430 | 0.1773 |
| max | 112 | 8 | 8.5159 | 8.4842 | 8.5563 | 0.0721 |
| max | 120 | 8 | 8.3988 | 8.3372 | 8.4768 | 0.1395 |
| max | 128 | 1 | 8.3124 | 8.3124 | 8.3124 | 0.0000 |
| min | 1 | 8 | 26.8695 | 24.5308 | 29.7405 | 5.2097 |
| min | 8 | 8 | 15.6053 | 14.0692 | 17.7576 | 3.6884 |
| min | 16 | 8 | 13.0478 | 12.3356 | 13.5607 | 1.2252 |
| min | 24 | 8 | 11.3495 | 10.5983 | 11.8620 | 1.2637 |
| min | 32 | 8 | 10.7107 | 10.4192 | 11.0694 | 0.6502 |
| min | 40 | 8 | 9.9964 | 9.7172 | 10.6464 | 0.9292 |
| min | 48 | 8 | 9.7139 | 9.3905 | 10.1267 | 0.7362 |
| min | 56 | 8 | 9.5521 | 9.2921 | 9.9986 | 0.7065 |
| min | 64 | 8 | 9.2945 | 9.0799 | 9.6185 | 0.5386 |
| min | 72 | 8 | 9.0451 | 8.9859 | 9.1319 | 0.1460 |
| min | 80 | 8 | 8.9284 | 8.8296 | 9.0002 | 0.1706 |
| min | 88 | 8 | 8.8561 | 8.7315 | 9.0204 | 0.2888 |
| min | 96 | 8 | 8.7830 | 8.6993 | 8.8972 | 0.1979 |
| min | 104 | 8 | 8.7332 | 8.6763 | 8.8225 | 0.1462 |
| min | 112 | 8 | 8.6743 | 8.6349 | 8.7289 | 0.0940 |
| min | 120 | 8 | 8.6081 | 8.5800 | 8.6219 | 0.0419 |
| min | 128 | 1 | 8.5771 | 8.5771 | 8.5771 | 0.0000 |
