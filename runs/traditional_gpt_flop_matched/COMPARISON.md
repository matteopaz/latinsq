# FLOP-Matched GPT vs Latin

The GPT baseline was matched to the estimated Latin training FLOP budget by increasing GPT optimizer updates/token exposure. Negative deltas mean GPT had lower validation loss than the Latin row.

| budget | k | GPT loss | Latin max | Latin min | GPT - max | GPT - min | GPT tokens | token ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000000 | 16 | 6.7748 | 8.8459 | 9.2553 | -2.0711 | -2.4805 | 6249984 | 15.9x |
| 1000000 | 64 | 5.8150 | 8.4150 | 8.7823 | -2.6000 | -2.9674 | 26654208 | 67.8x |
| 1000000 | 128 | 5.4682 | 8.4107 | 8.7392 | -2.9425 | -3.2710 | 59940864 | 152.4x |
| 10000000 | 16 | 6.5339 | 8.8147 | 9.0780 | -2.2808 | -2.5441 | 6156288 | 15.7x |
| 10000000 | 64 | 5.3870 | 8.2457 | 8.3987 | -2.8587 | -3.0117 | 24179712 | 61.5x |
| 10000000 | 128 | 4.8220 | 8.1407 | 8.3155 | -3.3187 | -3.4935 | 50330112 | 128.0x |

Result: under this FLOP-matched proxy, GPT is lower-loss than both Latin schedulers on all six matched rows. The comparison is still approximate because the FLOP model is analytic rather than measured kernel-level FLOPs, but it captures the main fairness correction: Latin spends much more compute per token, so GPT receives many more tokens at equal estimated compute.
