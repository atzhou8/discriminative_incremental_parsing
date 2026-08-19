import argparse
import string
import re
import stanza
import torch

import pandas as pd
import numpy as np

from pathlib import Path
from stanza.utils.conll import CoNLL
from torch.nn.utils.rnn import pad_sequence


from models.parser import Parser
from models.lc_crf_tagger import LinearChainCRFSuperTagger
from models.info_metrics import get_info_metrics
from utils import expand_sentences_to_word_rows, combine_multi_words

INFO_METRICS_TO_SAVE = [
    'kl_backward',
    'renyi_divergence_backward_2',
    'renyi_divergence_backward_3',
    'renyi_divergence_backward_4',
    'renyi_divergence_backward_5',
    'renyi_divergence_backward_6',  
]


def get_batch_from_word_rows(word_rows, batch_indices, device):
    """Batch out a slice from big dataframe to feed into parser."""
    sentences = [word_rows['SentenceTokenized'][i] for i in batch_indices]
    lengths = [len(sentence)+1 for sentence in sentences]
    cutoffs = [int(word_rows['word_pos'][i]) for i in batch_indices]

    return {
        'sentences': list(sentences), 
        'lengths': torch.tensor(lengths, dtype=torch.int64, device=device),
        'cutoffs': torch.tensor(cutoffs, dtype=torch.int64, device=device),
        'conditions': None
    }

def get_exact_metrics_for_batch(model, batch):
    # Compute information metrics for a prefix
    out_before = model.forward(
        sentences=[s.copy() for s in batch['sentences']],
        lengths=batch['lengths'],
        cutoffs=batch['cutoffs'],
        mask_last=True,
    )
    dist_before, before_sentences = out_before['crf'], out_before['cut_sentences']

    out_after = model.forward(
        sentences=[s.copy() for s in batch['sentences']],
        lengths=batch['lengths'],
        cutoffs=batch['cutoffs'],
    )
    dist_after, after_sentences = out_after['crf'], out_after['cut_sentences']

    metrics = get_info_metrics(dist_before, dist_after)
    before_sentences = [' '.join(sentence) for sentence in before_sentences]
    after_sentences = [' '.join(sentence) for sentence in after_sentences]
    return metrics, before_sentences, after_sentences


def add_info_metrics_all(
    model,
    items_path,
    batch_size=64,
    input_is_csv=False,
):    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    items_df = None
    if input_is_csv:
        items_df = pd.read_csv(items_path)
    else:
        items_df = pd.read_csv(items_path, sep='\t', header=None, names=['Sentence'])


    word_rows = expand_sentences_to_word_rows(items_df)
    num_rows = len(word_rows['Sentence'])
    with torch.no_grad():
        for metric in INFO_METRICS_TO_SAVE:
            word_rows[metric] = []
        for start in range(0, num_rows, batch_size):
            end = min(start + batch_size, num_rows)
            batch_indices = list(range(start, end))
            batch = get_batch_from_word_rows(word_rows, batch_indices, device)
            metrics, before_sentences, after_sentences = get_exact_metrics_for_batch(model, batch)

            for metric in INFO_METRICS_TO_SAVE:
                values = metrics[metric]
                for j, row_idx in enumerate(batch_indices):
                    word_rows[metric].append(values[j])

    df = pd.DataFrame(word_rows)
    combine_multi_words(df)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-v', '--version', type=int, required=True)
    parser.add_argument('-i', '--input_csv', default='data/phenomena/SAP/items_filler.csv')
    parser.add_argument('-o', '--output_csv', default=None)
    parser.add_argument('--ckpt', default='val', choices=['val', 'cutoff', 'last'])
    parser.add_argument('-m', '--model_type', default='parser', choices=['parser', 'tagger'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--input_is_csv', action='store_true')
    args = parser.parse_args()

    ckpt_dir = (
        Path('../lightning_logs')
        / args.name
        / f'version_{args.version}'
        / 'checkpoints'
    )
    if args.ckpt == 'last':
        best_ckpt = ckpt_dir / 'last.ckpt'
    else:
        best_ckpt = next(ckpt_dir.glob(f'best_{args.ckpt}_epoch=*.ckpt'))

    print(f'Loading checkpoint from {best_ckpt}')
    if args.model_type == 'parser':
        model = Parser.load_from_checkpoint(best_ckpt)
    elif args.model_type == 'tagger':
        model = LinearChainCRFSuperTagger.load_from_checkpoint(best_ckpt)
    else:
        raise TypeError('Unsupported model type')

    crf_type = getattr(model, 'crf_type', None)
    if crf_type is not None:
        print(f'Using CRF type: {crf_type}')

    output_csv = args.output_csv
    df = add_info_metrics_all(
        model=model,
        items_path=args.input_csv,
        batch_size=args.batch_size,
        input_is_csv=args.input_is_csv,
    )

    df.to_csv(output_csv)
    print(f'Wrote metrics to {output_csv}')
