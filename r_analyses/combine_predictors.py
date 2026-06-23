dataset = 'ClassicGP'

def combine_predictors(dataset):
    parser = pd.read_csv(f'../out/parser/items_{dataset}.parser.csv')
    gpt2 = pd.read_csv(f'../out/gpt2/items_{dataset}.gpt2.csv')
    roberta = pd.read_csv(f'../out/causal_roberta/items_{dataset}.word_surprisals.csv')
    synsurp = pd.read_csv(f'../out/synsurp/silver/items_{dataset}.synsurp.csv')
    ccg_kl = pd.read_csv(f'../out/synsurp/silver/items_{dataset}.synsurp_kl.csv')

    assert len(parser) == len(gpt2) == len(roberta) == len(synsurp) == len(ccg_kl)
    parser['gpt2_surp'] = gpt2['sum_surprisal']
    parser['length'] = gpt2['length']
    parser['logfreq'] = gpt2['logfreq']
    parser['roberta_surp'] = roberta['word_surprisal']
    parser['synsurp'] = synsurp['syn_surp']
    parser['ccg_kl'] = ccg_kl['kl']
    parser.to_csv(f'predictors/all_predictors.{dataset}.csv')

for dataset in ['ClassicGP', 'filler']:
    combine_predictors(dataset)
