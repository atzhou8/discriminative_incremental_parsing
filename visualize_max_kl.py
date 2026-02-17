import argparse
import os
import pandas as pd

from stanza.utils.conll import CoNLL
from tqdm import tqdm

def sentence_to_conllu_lines(sentence):
    '''
    Convert a stanza Sentence (from CoNLL.conll2doc) to a list of CoNLL-U lines.
    '''
    sent_dict = sentence.to_dict()  # list[dict] for tokens
    conll_sents = CoNLL.convert_dict([sent_dict])
    conll_sent = conll_sents[0]
    lines = []
    for cols in conll_sent:
        cols = [c if c is not None else '_' for c in cols]
        lines.append('\t'.join(cols))
    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--kl_csv',
        default='out/anchored/kl_amb.csv',
        help='Path to kl_amb.csv',
    )
    parser.add_argument(
        '--conllu_dir',
        default='out/anchored/parses',
        help='Directory containing masked-*.conllu and unmasked-*.conllu',
    )
    parser.add_argument(
        '--output',
        default='out/anchored/amb_max_cp.conllu',
        help='Output CoNLL-U file to write',
    )
    parser.add_argument(
        '--type',
        default='amb'
    )
    args = parser.parse_args()

    df = pd.read_csv(args.kl_csv, index_col=0)

    # KL columns and their increments
    kl_cols = [c for c in df.columns if c.startswith('kl+')]
    kl_increments = {c: int(c.split('+', 1)[1]) for c in kl_cols}

    with open(args.output, 'w', encoding='utf-8') as out_f:
        writeln = lambda x: out_f.write(x + '\n')

        row_indices = range(len(df))
        for row_idx in tqdm(row_indices):
            row = df.iloc[row_idx]
            cp = int(row['cp'])
            condition = row['condition']
            sentence_txt = row['sentence']

            # Best KL column
            best_kl_col = row[kl_cols].astype(float).idxmax()
            best_kl_val = float(row[best_kl_col])
            inc = kl_increments[best_kl_col]
            max_cp = cp + inc
            perp_mask = row[f'perp_mask+{inc}']
            perp_unmask = row[f'perp_unmask+{inc}']

            writeln(f'# condition = {condition}')
            writeln(f'# text = {sentence_txt}')
            writeln(f'# cp = {cp}')
            writeln(f'# max_cp = {max_cp}')
            writeln(f'# diff={inc}')
            writeln(f'# kl = {best_kl_val:.4f}')
            writeln(f'# perp = {perp_mask}')

            masked_path = os.path.join(args.conllu_dir, f'{args.type}_masked-{max_cp}.conllu')
            masked_doc = CoNLL.conll2doc(masked_path)
            masked_sent = masked_doc.sentences[row_idx]
            for line in sentence_to_conllu_lines(masked_sent):
                writeln(line)
            writeln('')

            writeln(f'# text = {sentence_txt}')
            writeln(f'# perp = {perp_unmask}')
            unmasked_path = os.path.join(args.conllu_dir, f'{args.type}_unmasked-{max_cp}.conllu')
            unmasked_doc = CoNLL.conll2doc(unmasked_path)
            unmasked_sent = unmasked_doc.sentences[row_idx]
            for line in sentence_to_conllu_lines(unmasked_sent):
                writeln(line)
            writeln('')
            writeln('#' * 80)
