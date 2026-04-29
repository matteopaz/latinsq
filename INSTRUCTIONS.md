# The Experiment

Experimenting with a new LLM architecture composed of submodels, which is intended to enable peer-to-peer edge AI on consumer hardware with minimal interconnects.

## The Architecture

First, let there be K transformers, which we call the "submodels". All submodels are much smaller than a traditional LLM, and have very little depth each (1-2 layers). In addition, they have very few attention heads (2-4), and carry a small d_model (<1024). They are analogous to a single transformer block.

To define the flow of data through the model, we use a latin square of order K. For each of the K copies of the data, 

At step 0, the input sequence is transformed via a simple embedding matrix. These are tied to the head matrix.

At step 1, K copies of the data are made, and paired to a column of the latin square.

For the next K steps, each copy of the data flows through the subsequent row of its latin square column. I.e., the data flow traces out a unique permutation of the K submodels. 

For the next step, the embedded sequence is deprojected using weights tied to the inital embedding matrix by transpose.

Finally, we are left with K sets of vocab output logits. To produce the final prediction, average probabilities across the K, with temperature 1.

## Training / Details

Train on the entirety of wikitext for 2 epochs. Use a vocab size of 32K.
Use the AdamW optimizer with reasonable weight decay.
Optimize training.

## Scans

Scan across K=16,64,128 with total model parameters (sum across all submodels) (exclude embed and head) of 1M and 10M.

Perform the above two times. One time using the min-entropy (cyclic) latin square as the scheduler, and one time using the max entropy. The generator for these squares is already implemented. All plots should have traces for these two cases.

# Deliverables

Training curves for all configurations of K and params.

Plot: Final held-out validation loss (y) vs K (x), separate traces for total model size / scheduler.

Table: Same as the above but tabular

Ensembling Analysis:: Answer these questions, and if any of the scanned parameters affects the answer.
1. Are some ensemble members (one of the final K sequences) consistently more certain? 
2. How much better is the ensembled answer than the average individual answer?
3. On average, how dissimilar are each of the ensemble members predictions?

Writeup::
Any additional findings e.g. difficulties with gradient flow, bottlenecks which slow down training, instabilities