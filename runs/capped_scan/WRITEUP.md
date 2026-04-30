# Latin Square Ensemble Experiment

## Configuration

- Dataset: Wikitext-103 raw validation/training split via `Salesforce/wikitext`, BPE vocab size 32,000.
- Scan: K in {16, 64, 128}, total submodel parameters in {1M, 10M}, schedulers in {min, max}.
- Training: AdamW, weight decay 0.1, tied token embedding/output projection, causal attention, two epochs unless a run was explicitly capped by CLI arguments.
- Run scope: the literal full two-epoch sweep is computationally large for the K=64/128 cells, so this report uses bounded runs. The exact train/eval caps and effective batch sizes are recorded below and in `summary.csv`.

## Run Scope

k,total_params,scheduler,max_train_batches,max_eval_batches,effective_train_batch_size,effective_eval_batch_size
16,1000000,max,512,128,16,4
64,1000000,max,512,128,8,1
128,1000000,max,128,64,2,1
16,1000000,min,512,128,16,4
64,1000000,min,512,128,8,1
128,1000000,min,128,64,2,1
16,10000000,max,512,128,16,4
64,10000000,max,128,64,8,1
128,10000000,max,128,64,2,1
16,10000000,min,512,128,16,4
64,10000000,min,128,64,8,1
128,10000000,min,128,64,2,1


## Results Table

k,total_params,scheduler,d_model,ensemble_loss,avg_individual_loss,ensemble_gain,member_certainty_mean,member_certainty_std,member_certainty_range,prediction_dissimilarity_kl
16,1000000,max,72,7.395676,20.976206,13.580530,0.265480,0.153693,0.539860,2.544290
64,1000000,max,36,7.674487,18.398823,10.724336,0.318032,0.293856,0.956781,3.466085
128,1000000,max,24,8.879786,18.533861,9.654076,0.321955,0.288058,0.906731,3.590727
16,1000000,min,72,7.407371,20.478008,13.070637,0.254625,0.177713,0.615073,2.475890
64,1000000,min,36,7.650863,17.464584,9.813721,0.334698,0.243576,0.841295,3.026900
128,1000000,min,24,9.348959,17.996723,8.647764,0.273157,0.205548,0.765241,3.124103
16,10000000,max,228,7.267820,31.635630,24.367810,0.300813,0.236396,0.911641,2.572098
64,10000000,max,114,8.609490,31.540836,22.931347,0.376909,0.318926,0.929275,3.834614
128,10000000,max,80,9.190512,29.618681,20.428169,0.459649,0.328244,0.916643,4.295700
16,10000000,min,228,7.276775,29.941314,22.664539,0.272916,0.156555,0.524457,2.441732
64,10000000,min,114,8.745440,30.133285,21.387844,0.379827,0.291825,0.945452,3.569725
128,10000000,min,80,9.480752,28.212063,18.731311,0.393486,0.301870,0.922198,3.790919


## Validation Loss

See `validation_loss_vs_k.png` for final held-out validation loss versus K with separate traces for scheduler and model-size settings.

## Training Curves

See `training_curves_all.png` for all training-loss traces and `curve_k*_p*_*.png` for individual run plots.

## Ensembling Analysis

- Best final held-out ensemble loss: 7.267820 at K=16, params=10000000, scheduler=max.
- Certainty consistency: member-certainty ranges averaged by scheduler were {"max": 0.860155217970411, "min": 0.7689527080704769}. Larger ranges indicate that some final sequence members are consistently more certain.
- Ensemble benefit: average individual CE minus ensemble CE averaged by parameter budget was {"1000000": 10.915177146089263, "10000000": 21.751836680147488}. Positive values mean ensembling improved held-out CE over the average individual member.
- Prediction dissimilarity: mean KL-to-ensemble by scheduler was {"max": 3.383918971133729, "min": 3.0715446236232915}. Higher values indicate more diverse member predictions.

## Additional Findings

- The dominant memory and runtime cost is materializing logits for every ensemble member with shape `[batch, K, seq_len, vocab]`; K=128 with a 32K vocabulary is the bottleneck.
- Training uses an exact chunked hidden-state ensemble loss to avoid materializing the full training-time `[batch, K, seq_len, vocab]` logits tensor, but total compute still scales with K and vocabulary size.
- The implementation remains resumable: result JSON files are reused, and summary/report artifacts are regenerated from all completed runs.
