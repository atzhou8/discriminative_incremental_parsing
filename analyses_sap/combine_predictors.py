import pandas as pd
import numpy as np

dataset = 'ClassicGP'

def combine_predictors(dataset):
    parser = pd.read_csv(f'../out/parser/items_{dataset}.parser.csv')
    gpt2 = pd.read_csv(f'../out/gpt2/items_{dataset}.gpt2.csv')
    roberta = pd.read_csv(f'../out/causal_roberta/items_{dataset}.word_surp.csv')
    # synsurp = pd.read_csv(f'../out/synsurp/silver/items_{dataset}.synsurp.csv')
    # ccg_kl = pd.read_csv(f'../out/synsurp/silver/items_{dataset}.synsurp_kl.csv')

    assert len(parser) == len(gpt2) == len(roberta) # == len(synsurp) == len(ccg_kl)
    parser['gpt2_surp'] = gpt2['sum_surprisal']
    parser['length'] = gpt2['length']
    parser['logfreq'] = gpt2['logfreq']
    parser['roberta_surp'] = roberta['word_surprisal']
    if dataset == 'ClassicGP':
        supertag = pd.read_csv(f'../out/ccg_ClassicGP.csv')
        parser['supertag_kl'] = supertag['kl_backward']
    # parser['synsurp'] = synsurp['syn_surp']
    # parser['ccg_kl'] = ccg_kl['kl']
    parser['exp_kl'] = np.exp(parser['kl_backward'])
    parser['quad_kl'] = parser['kl_backward']**2
    parser['RI'] = roberta['word_surprisal'] - parser['kl_backward']
    parser.to_csv(f'predictors/all_predictors.{dataset}.csv')

for dataset in ['ClassicGP', 'filler']:
    combine_predictors(dataset)
