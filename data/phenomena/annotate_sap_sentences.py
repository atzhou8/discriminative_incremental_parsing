"""
SAP phenomena are given as raw sentences. This script tokenizes these 
sentences according to UD, as that is what we did during training. Critical
points are manually corrected since we have so few examples and most match 
anyway, but will probably need to think of something more clever in the 
future. 

Only mismatch I can see between UD tokenization and SAP indexing is one instance
of "show's" in item 22 after the critical point, and the leading commas
before the critical point in the unambiguous NP/Z examples

"""

import pandas as pd
import stanza

stanza.download("en", processors="tokenize", verbose=False)
tok = stanza.Pipeline("en", processors="tokenize", tokenize_no_ssplit=True)

phen_dir = 'SAP/sap_garden_paths_unamb.csv'

df = pd.read_csv(phen_dir)
sentences = df['sentence'].tolist()
cps = df['critical_pos'].tolist()
ud_splits = tok(sentences)

ud_sentences = [' '.join([w.text for w in sentence.words]) for sentence in ud_splits.sentences]
cp_words = [s.split(' ')[cp-1] for s, cp in zip(ud_sentences, cps)]

crit_idx = df.columns.get_loc('critical_pos')
df.insert(crit_idx + 1, 'critical_word', cp_words)
df['sentence'] = ud_sentences
df.to_csv(phen_dir, index=False)

