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
from models.info_metrics import get_info_metrics, uniform_dist_like, recovered_dist_like

INFO_METRICS_TO_SAVE = [
    'kl_backward',
    'renyi_divergence_backward_2',
    'renyi_divergence_backward_3',
    'renyi_divergence_backward_4',
    'renyi_divergence_backward_5',
    'renyi_divergence_backward_6',  
]
nlp = stanza.Pipeline(
    lang='en',
    processors='tokenize,mwt',
    tokenize_no_ssplit=True,
    use_gpu=torch.cuda.is_available()
)

def split_by_space_and_hyphen(text):
    pattern = r'\s+|(?<=-)'
    return [word for word in re.split(pattern, text) if word]

def get_subtokens(word):
    doc = nlp(word)
    tokens = doc.sentences[0].tokens
    return [w.text for tok in tokens for w in tok.words]

def create_word_rows_for_sentence(sentence):
    words = split_by_space_and_hyphen(sentence)

    word_rows = []
    for word in words:
        word_rows.append({
            'unsplit': word,
            'split': get_subtokens(word)
        })
    return word_rows

# QUOTE_CHAR = '"'

# def _extend_last_row(word_rows, text):
#     if not word_rows:
#         word_rows.append({'unsplit': text, 'split': [text]})
#         return
#     word_rows[-1]['unsplit'] += text
#     if word_rows[-1]['split']:
#         word_rows[-1]['split'][-1] += text
#     else:
#         word_rows.append({'unsplit': text, 'split': [text]})


# def _append_punct_row(word_rows, text):
#     if not word_rows:
#         word_rows.append({'unsplit': text, 'split': [text]})
#     else:
#         word_rows[-1]['unsplit'] += text
#         word_rows[-1]['split'].append(text)


# def create_word_rows_for_sentence(sentence):
#     """Expand out a sentence into tokenization by punctuation."""
#     doc = nlp(sentence)
#     tokens = doc.sentences[0].tokens  # type: ignore
#     word_rows = []
#     pending_opening_quote = ''
#     in_compound = False

#     for idx, token in enumerate(tokens):
#         if token.text == QUOTE_CHAR:
#             prev_tok = tokens[idx - 1] if idx > 0 else None
#             next_tok = tokens[idx + 1] if idx + 1 < len(tokens) else None

#             attaches_to_prev = (
#                 prev_tok is not None and prev_tok.end_char == token.start_char
#             )
#             attaches_to_next = (
#                 next_tok is not None and next_tok.start_char == token.end_char
#             )

#             if attaches_to_prev:
#                 # no space before the quote -> it closes the previous word
#                 _extend_last_row(word_rows, token.text)
#                 pending_opening_quote = ''
#             elif attaches_to_next:
#                 # no space after the quote -> it opens the next word
#                 pending_opening_quote += token.text
#             else:
#                 # isolated quote (space on both sides) -> treat as its own token
#                 _append_punct_row(word_rows, token.text)
#                 pending_opening_quote = ''
#             continue

#         if token.text in string.punctuation or in_compound:
#             _append_punct_row(word_rows, token.text)
#             in_compound = token.text == '-'
#             continue

#         word_rows.append({
#             'unsplit': pending_opening_quote + token.text,
#             'split': [word.text for word in token.words],
#         })
#         pending_opening_quote = ''

#     return word_rows

def expand_items_to_word_rows(items_df):
    """Expands items dataframe from SAP dataset that contains only sentences
    to instead contain one token per row."""
    base_columns = list(items_df.columns)
    new_columns = [
        'SentenceTokenized',
        'word_pos', 
        'EachWord',
        'WordTokens',
        'SentenceStart',
        'WordStart', 
        'IsMultiWord',
        'WordStart',
    ]
    output_dict = {column: [] for column in base_columns + new_columns}

    for idx, (_, row) in enumerate(items_df.iterrows()):
        sentence_word_rows = create_word_rows_for_sentence(row['Sentence'])
        tokenized_sentence = [item for word_row in sentence_word_rows for item in word_row['split'] ]
        pos = 1
        for i, word_row in enumerate(sentence_word_rows):
            each_word = word_row['unsplit']
            word_tokens = word_row['split']
            for j, token in enumerate(word_tokens):
                for column in base_columns:
                    output_dict[column].append(row[column])
                output_dict['SentenceTokenized'].append(tokenized_sentence)
                output_dict['word_pos'].append(pos)
                output_dict['EachWord'].append(each_word)
                output_dict['WordTokens'].append(token)
                output_dict['SentenceStart'].append(i==0 and j==0)
                output_dict['WordStart'].append(j==0)
                output_dict['IsMultiWord'].append(len(word_tokens)>1)
                pos += 1

    return output_dict

def combine_multi_words(word_rows):
    """After writing metrics, sum metric values over tokenized words 
    and delete non-initial tokens"""
    indices_to_drop = []
    deleted_in_curr_sent = 0
    for word_index, row in word_rows.iterrows():
        if row['SentenceStart']:
            deleted_in_curr_sent = 0
        word_rows.loc[word_index, 'word_pos'] -= deleted_in_curr_sent
        if row['IsMultiWord'] and row['WordStart']:
            found_full_word = False
            increment = 1
            word_length = 1
            while not found_full_word and word_index+increment < len(word_rows):
                next_row = word_rows.iloc[word_index+increment]
                if next_row['WordStart']:
                    found_full_word = True
                else:
                    indices_to_drop.append(word_index+increment)
                    deleted_in_curr_sent += 1
                    word_length += 1
                    increment += 1 
                    for metric in INFO_METRICS_TO_SAVE:
                        word_rows.loc[word_index, metric] += next_row[metric]


            
            for metric in INFO_METRICS_TO_SAVE:
                word_rows.loc[word_index, metric] /= 1

    word_rows.drop(index=indices_to_drop, inplace=True)
    word_rows.reset_index(drop=True, inplace=True)

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

def get_metrics_for_batch(model, batch):
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
    output_path,
    batch_size=64,
):    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    items_df = pd.read_csv(items_path, sep='\t', header=None, names=['Sentence'])
    word_rows = expand_items_to_word_rows(items_df)
    num_rows = len(word_rows['Sentence'])
    with torch.no_grad():
        for metric in INFO_METRICS_TO_SAVE:
            word_rows[metric] = []
        for start in range(0, num_rows, batch_size):
            end = min(start + batch_size, num_rows)
            batch_indices = list(range(start, end))
            batch = get_batch_from_word_rows(word_rows, batch_indices, device)
            metrics, before_sentences, after_sentences = get_metrics_for_batch(model, batch)

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
    args = parser.parse_args()

    ckpt_dir = (
        Path('lightning_logs')
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
        output_path=output_csv,
        batch_size=args.batch_size,
    )

    df.to_csv(output_csv)
    print(f'Wrote metrics to {output_csv}')
