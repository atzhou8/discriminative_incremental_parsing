# Syntactic Belief Update as the Driver of Garden Path Processing Difficulty

This repository holds code for the paper [Syntactic Belief Update
as the Driver of Garden Path Processing Difficulty](https://arxiv.org/).

Checkpoints for our parser model and baselines can be downloaded [here](https://drive.google.com/).

Large `brms` model files used for our statistical analyses can be downloaded [here](https://drive.google.com/).

## Obtaining Syntactic Belief Update metrics for arbitrary text data
Download our pretrained parser checkpoint to `lightning_logs/` or train a new model using
```
python train_parser.py NAME -train TRAIN_DIR -val VAL_DIR 
```
where `TRAIN_DIR` and `VAL_DIR` are expected to be treebanks in the CoNLL-U format.

Then, run `compute_metrics_for_sentences.py` to compute SBU metrics given a text file that contains one sentence per line:
```
python compute_metrics_for_sentences.py -n NAME -i sentences.txt -o output.csv
```

This will output a csv with a column for each word, as well as a column for each SBU metric (`kl_backward` for KL Divergence, and `renyi_divergence_backward_n` for Rényi Divergences with different parameters $\alpha$)

## Reproducing results reported in the paper
Download pretrained models to `lightning_logs\`

Download SAP benchmark items csvs to `data\phenomena\SAP\items\`

### Compute metrics:
1. Compute SBU metrics:
    ```
    python compute_metrics_for_items.py \
    -n parser \
    -v 0 \
    -i data/phenomena/SAP/items/items_ClassicGP.pivot.csv \
    -o out/parser/items_ClassicGP.parser.csv

    python compute_metrics_for_items.py \
    -n parser \
    -v 0 \
    -i data/phenomena/SAP/items/items_filler.pivot.csv \
    -o out/parser/items_filler.parser.csv
    ```
2. Compute syntactic surprisal metrics:
    ```
    python compute_synsurp_for_items.py \
    -n parser \
    -v 0 \
    -i data/phenomena/SAP/items/items_ClassicGP.pivot.csv \
    -o out/parser/items_ClassicGP.synsurp.csv

    python compute_synsurp_for_items.py \
    -n parser \
    -v 0 \
    -i data/phenomena/SAP/items/items_filler.pivot.csv \
    -o out/parser/items_filler.synsurp.csv
    ```
3. Compute causal RoBERTa surprisal metrics:
    ```
    python compute_causal_roberta_surprisal_for_items.py \
    -i data/phenomena/SAP/items/items_ClassicGP.pivot.csv \
    -o data/phenomena/SAP/items/items_ClassicGP.word_surp.csv \
    -m lightning_logs/causal_roberta/checkpoint-1200

    python compute_causal_roberta_surprisal_for_items.py \
    -i data/phenomena/SAP/items/items_filler.pivot.csv \
    -o data/phenomena/SAP/items/items_filler.word_surp.csv \
    -m lightning_logs/causal_roberta/checkpoint-1200
    ```

### Run R analyses
cd into the `r_analyses` directory to run statistical analyses
1. Prepare metrics for R analyses
    ```
    cd r_analyses
    python get_mergedRT.py predictors/ClassicGardenPathSet.csv predictors/ClassicGardenPathSet_merged.csv
    python get_mergedRT.py predictors/Fillers.csv predictors/Fillers_merged.csv
    python combine_metrics.py
    ```
2. Fit linear mixed effects models for predicting reading times
    ```
    Rscript fit_mergedRT_lms_fillers.R
    Rscript fit_mergedRT_rois_fillers.R 
    ```
    These save models to `r_analyses/results/rt_models/mergedRT/{filler, roi}/eachword/` by default.

3. Fit Bayesian mixed effects models for estimating garden path effect.

    (**Warning**: these models take a very long time to sample, we recommend downloading prefit brms models [here](drive.google.com))
    ```
    Rscript fit_mergedEOI_brms_from_empirical.R
    Rscript fit_mergedEOI_brms_from_mergedlms.R \
        [0-10] \ 
        results/rt_models/filler/eachword/ \
        results/brms_eoi_models/mergedRT/filler/
    Rscript fit_mergedEOI_brms_from_mergedlms.R \
        [0-10] \ 
        results/rt_models/roi/eachword/ \
        results/brms_eoi_models/mergedRT/roi/
    ```
    The [0-10] argument of `fit_mergedEOI_brms_from_mergedlms.R` determines the index of the rt_model to run in the specified directory, allowing for easy batch processing on high-performance computing systems.

    The brms for the the mergedlms are not guaranteed to converge with the default settings. We experimented by increasing the number of iterations, acceptance probability, and maximum tree depth if the Rhat for any coefficient of the fit model was greater than 1.05. For some models we needed to simplify the effect structure by removing the (1 | participant) intercept term. 

4. Extract construction-level garden path effects
    ```
    Rscript get_EOIs_from_brms.R results/brms_eoi_models/mergedRT/roi/ results/EOIs/roi_eois.csv
    Rscript get_EOIs_from_brms.R results/brms_eoi_models/mergedRT/filler/ results/EOIs/filler_eois.csv
    ```

5. Extract item-level garden path effect correlations
    ```
    Rscript results/brms_eoi_models/mergedRT/roi/ results/correlations/roi_raw.csv results/correlations/roi_corr.csv
    Rscript results/brms_eoi_models/mergedRT/filler/ results/correlations/filler_raw.csv results/correlations/filler_corr.csv
    ```

6. Make plots following notebook
    ```
    jupyter notebook make_plots_for_papers.ipynb
    ```