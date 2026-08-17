import re

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

lang = 'English'
metrics_dir = Path(f'../metrics/MECO/{lang}/')
freqs_dir = Path('../data/wordfreqs/freqs_coca.csv')
freqs_coca = pd.read_csv(freqs_dir)

def get_logfreq_for_word(word):
    if word in freqs_coca['word'].tolist():
        freq = int(freqs_coca.loc[freqs_coca['word']==word]['count'].iloc[0])
    else:
        print(f'{word} not found in corpus, using log(1) as freq')
        freq = 1
    return np.log(freq)


logfreqs = []
lengths = []
metrics = pd.read_csv(metrics_dir / f'all_predictors.csv')
for i, row in tqdm(metrics.iterrows(), total=len(metrics)):
    word = row['EachWord'].lower()
    word = re.sub(r"[^\w\s]", "", word)
    logfreqs.append(get_logfreq_for_word(word))
    lengths.append(len(word))

metrics['logfreq'] = logfreqs
metrics['length'] = lengths
metrics.to_csv(metrics_dir / f'all_predictors.csv')