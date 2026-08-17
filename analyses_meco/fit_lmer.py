import os

import polars as pl
import pandas as pd
import numpy as np

from pymer4.models import lmer, glmer
from pymer4.io import save_model, load_model

from scipy.stats import norm

from pathlib import Path
from tqdm import tqdm



# Fitting info
measures_of_interest = [
    'firstrun.reg.out', # was there a leftward regression?
    'firstrun.dur', # Gaze duration e.g. forward reading time
    'firstrun.gopast.sel', # if there was a regression, how long was spent on word
    'regressive.gopast', # if there was a regression, how long was the regression (firstrun.gopast.sel - firstrun.gopast)
]

# Distribution family for each measure. 'firstrun.reg.out' is a binary
# indicator, so it's fit as a logistic (binomial) mixed model rather than
# a Gaussian one.
measure_families = {
    'firstrun.reg.out': 'binomial',
    'firstrun.dur': 'gaussian',
    'firstrun.gopast.sel': 'gaussian',
    'regressive.gopast': 'gaussian',
}

# lme4 control strings differ between lmer() and glmer() fits
# (lmerControl vs glmerControl), so keep them keyed by family.
control_strings = {
    'gaussian': 'optimizer = "bobyqa", optCtrl = list(maxfun = 20000), lmerControl(calc.derivs = FALSE)',
    'binomial': 'optimizer = "bobyqa", optCtrl = list(maxfun = 100000), glmerControl(calc.derivs = FALSE)',
}

# metrics_of_interest = [
#     'RI_tagger',
#     'RI_parser',
#     'tagger_renyi_divergence_backward_2',
#     'tagger_renyi_divergence_backward_3',
#     'tagger_renyi_divergence_backward_4',
#     'renyi_divergence_backward_2',
#     'renyi_divergence_backward_3',
#     'renyi_divergence_backward_4',
# ]

metrics_of_interest = [
    'word_surprisal',
    'kl_backward',
    'tagger_kl_backward'
]

baseline_predictors = [
    'logfreq',
    'length',
    'word_pos',
]
random_groups = [
    'subid',
]
et_dir = Path('../data/phenomena/MECO/my_data/english_measures.csv')
metrics_dir = Path(f'../metrics/MECO/English/')
out_dir = Path('./lms')

# Load and join data
et_data = pd.read_csv(et_dir)
passage_data = pd.read_csv(metrics_dir / f'all_predictors.csv')
all_data = pd.merge(
    et_data, passage_data, 
    left_on=['trialid', 'wordnum'], 
    right_on=['passage', 'wordnum'], 
    how='inner', 
)
# z-norm
all_data = pl.DataFrame(all_data)
all_data = all_data.with_columns([
    ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c)
    for c in metrics_of_interest + baseline_predictors
])
diff = len(et_data) - len(all_data)
print(f'{diff} ET trials dropped in merge; ; remaining {len(all_data)}.')

# Make lagged predictors
group_cols = ['subid', 'passage']
sort_cols = group_cols + ['wordnum']
lags = range(1, 3)
all_data = all_data.sort(sort_cols)
lagged_columns = [
    pl.col(col).shift(k).over(group_cols).alias(f"{col}_lag{k}")
    for col in metrics_of_interest + baseline_predictors
    for k in lags
]
all_data = all_data.with_columns(lagged_columns)
lagged_predictors = [f'{p}_lag{k}' for p in metrics_of_interest + baseline_predictors for k in lags]
all_data_cleaned = all_data.drop_nulls(subset=lagged_predictors)
diff = len(all_data) - len(all_data_cleaned)
all_data = all_data_cleaned
print(f'{diff} ET trials dropped from lags; remaining {len(all_data)}.')

# Get CV folds over passage, wordnum
nfolds = 10
pw_combos = all_data.select(['passage', 'wordnum']).unique()
pw_combos = pw_combos.sample(fraction=1.0, shuffle=True, seed=123)
pw_combos = pw_combos.with_columns(
    (pl.int_range(0, pl.len()) % nfolds).alias('fold')
)
all_data = all_data.join(pw_combos, on=['passage', 'wordnum'], how='left')
all_data.write_csv('lms/folds/full_data.csv')



# Loglikelihood function
def get_loglik(model, test_data, measure_of_interest, family='gaussian'):
    y_test = test_data[measure_of_interest].to_numpy()

    if family == 'binomial':
        # Predicted probabilities on the response scale (not logits).
        # glmer.predict() exposes this as `type_predict`, not `type`
        # (it forwards internally as type=type_predict, so passing
        # `type=` directly collides with that and raises a TypeError).
        preds = model.predict(test_data, type_predict='response')
        eps = 1e-9
        p = np.clip(preds, eps, 1 - eps)
        log_liks = y_test * np.log(p) + (1 - y_test) * np.log(1 - p)
    else:
        preds = model.predict(test_data)
        sigma = model.result_fit_stats['sigma'][0]
        log_liks = norm.logpdf(y_test, loc=preds, scale=sigma)

    return np.mean(log_liks)


# Baseline model setup
baseline_predictors  += [f'{pred}_lag{k}' for pred in ['logfreq', 'length'] 
                                        for k in lags]
baseline_formula = ' + '.join(baseline_predictors)
random_baseline_formula = ' + '.join(
    [f'(1 | {group})' for group in random_groups]
)
# baseline_transforms = {p:'scale' for p in baseline_predictors}

# Fit models
logliks = {
    'measure': [],
    'model': [],
    'fold': [],
    'loglik': []
}
for fold in tqdm(range(nfolds), desc='Fold'):
    train = all_data.filter(pl.col('fold') != fold)
    test = all_data.filter(pl.col('fold') == fold)
    for measure in tqdm(measures_of_interest, leave=False, desc='Measure'):
        family = measure_families[measure]
        ModelClass = glmer if family == 'binomial' else lmer
        model_kwargs = {'family': 'binomial'} if family == 'binomial' else {}

        # Drop trials where measure is nan
        curr_train = train.drop_nans(subset=[measure]).drop_nulls(subset=[measure])
        curr_test = test.drop_nans(subset=[measure]).drop_nulls(subset=[measure])
        curr_formula = f'{measure} ~ {baseline_formula} + {random_baseline_formula}'
        print('Baseline', curr_formula)

        # Fit baselines
        # baseline_dir = out_dir / f'f{fold}_{measure}_baseline.joblib'
        # if os.path.exists(baseline_dir):
        #     model = load_model(baseline_dir)
        # else:
        model = ModelClass(
            formula=curr_formula,
            data=curr_train,
            **model_kwargs
        )
        # model.set_transforms(baseline_transforms)
        model.fit(control=control_strings[family])
        # save_model(model, out_dir / f'f{fold}_{measure}_baseline.joblib')

        # Record loglik
        logliks['measure'].append(measure)
        logliks['model'].append('baseline')
        logliks['fold'].append(fold)
        logliks['loglik'].append(get_loglik(model, curr_test, measure, family=family))
        pd.DataFrame(logliks).to_csv('baseline_logliks.csv')

        # Fit metrics
        for predictor in tqdm(metrics_of_interest, leave=False, desc='Metric'):
            predictors = [f'{predictor}_lag{k}' for k in lags]
            predictors += [predictor]
            predictor_formula = ' + '.join(predictors)
            random_formula = ' + '.join(f'(1 + {predictor_formula} | {group})' 
                                        for group in random_groups)
            # pred_transforms = {p:'scale' for p in predictors}
            curr_formula = f'{measure} ~ {predictor_formula} + {baseline_formula} + {random_formula}'

            model = ModelClass(
                formula=curr_formula,
                data=curr_train,
                **model_kwargs
            )
            # model.set_transforms(baseline_transforms | pred_transforms)
            model.fit(control=control_strings[family])
            # save_model(model, out_dir / f'f{fold}_{measure}_{predictor}.joblib')

            # Record loglik
            logliks['measure'].append(measure)
            logliks['model'].append(predictor)
            logliks['fold'].append(fold)
            logliks['loglik'].append(get_loglik(model, curr_test, measure, family=family))
            pd.DataFrame(logliks).to_csv('baseline_logliks.csv')