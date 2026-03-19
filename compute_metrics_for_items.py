import argparse
import string
import stanza
import torch

import pandas as pd
import numpy as np

from pathlib import Path
from stanza.utils.conll import CoNLL
from torch.nn.utils.rnn import pad_sequence


from model.parser import Parser
from model.utils import get_info_metrics, uniform_dist_like, recovered_dist_like

info_metrics_to_save = [
    'kl_forward',
    # 'kl_root_shifted',
    'kl_backward',
    'kl_symmetric',
    'js_geo',
    'cross_entropy_forward',
    'cross_entropy_backward',
    'renyi_divergence_forward_5',
    'renyi_divergence_backward_5',
    'renyi_divergence_symmetric_5',   
]
nlp=stanza.Pipeline(lang='en', processors='tokenize')


def create_word_rows_for_sentence(sentence):
    doc = nlp(sentence)
    word_rows = [] 
    absorb_next = False
    for token in doc.sentences[0].tokens:
        # i.e. 'don't': ["do", "n't"]
        # TODO: add a check for sentences that begin with punctuation
        if token.text in string.punctuation or absorb_next:
            word_rows[-1]['unsplit'] += token.text
            word_rows[-1]['split'] += [token.text]
            absorb_next = False
            if token.text == '-': # special rule for compounds
                absorb_next = True
        else:
            word_rows.append(
                {'unsplit': token.text,
                'split': [word.text for word in token.words]}
            )
    return word_rows
    
def expand_items_to_word_rows(items_df, sentence_gold_indices=None):
    base_columns = list(items_df.columns)
    new_columns = [
        'SentenceTokenized',
        'WordPosition', 
        'EachWord',
        'WordTokens', 
        'IsMultiWord',
        'WordStart',
        'SentenceStart',
        'SentenceIndex', 
        'GoldTree',
        'RecoveredTree',
        'BeforeSentence',
        'AfterSentence'
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
                output_dict['WordPosition'].append(pos)
                output_dict['EachWord'].append(each_word)
                output_dict['WordTokens'].append(token)
                output_dict['WordStart'].append(j==0)
                output_dict['SentenceStart'].append(i==0 and j==0)
                output_dict['IsMultiWord'].append(len(word_tokens)>1)
                output_dict['SentenceIndex'].append(idx)
                output_dict['GoldTree'].append(
                    np.nan if sentence_gold_indices is None else sentence_gold_indices[idx]
                )
                output_dict['RecoveredTree'].append(np.nan)
                output_dict['BeforeSentence'].append(np.nan)
                output_dict['AfterSentence'].append(np.nan)
                pos += 1

    
    return output_dict

def combine_multi_words(word_rows, has_gold=False):
    indices_to_drop = []
    num_deleted = 0 # decrement pos by this amount
    for word_index, row in word_rows.iterrows():
        if row['SentenceStart']:
            num_deleted = 0
        word_rows.loc[word_index, 'WordPosition'] -= num_deleted
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
                    num_deleted += 1
                    word_length += 1
                    increment += 1 
                    for metric in info_metrics_to_save:
                        word_rows.loc[word_index, metric] += next_row[metric]
                        if has_gold:
                            word_rows.loc[word_index, f'{metric}_gold'] += next_row[f'{metric}_gold']

            
            for metric in info_metrics_to_save:
                word_rows.loc[word_index, metric] /= word_length
                if has_gold:
                    word_rows.loc[word_index, f'{metric}_gold'] /= word_length
                
    
    word_rows.drop(index=indices_to_drop, inplace=True)
    word_rows.reset_index(drop=True, inplace=True)

def load_gold_trees(word_rows, gold_path):
    """Super narrow method to load the gold_trees specifically for 
    ambiguous garden path items"""
    trees = CoNLL.conll2doc(gold_path).sentences
    trees = [[0] + [w.head for w in tree.words] for tree in trees]
    word_rows['GoldTree'] = word_rows['GoldTree'].astype(object)

    for index, row in word_rows.iterrows():
        if row['ambiguity'] == 'ambiguous':
            sentence_index = row['sentence_index']
            word_rows.at[index, 'GoldTree'] = trees[sentence_index]

def get_batch_from_word_rows(word_rows, id_start, id_end, device):
    sentences = [word_rows['SentenceTokenized'][i] for i in range(id_start, id_end)]
    lengths = [len(sentence)+1 for sentence in sentences]
    cutoffs = [int(word_rows['WordPosition'][i]) for i in range(id_start, id_end)]
    gold_trees = [word_rows['GoldTree'][i] for i in range(id_start, id_end)]
    if not type(gold_trees[0]) is list:
        gold_trees = None
    else:
        gold_trees = [torch.tensor(tree) for tree in gold_trees]
        gold_trees = pad_sequence(gold_trees, batch_first=True, padding_value=0).long()

    return {
        'sentences': list(sentences), 
        'gold_trees': gold_trees, 
        'lengths': torch.tensor(lengths, dtype=torch.int64, device=device),
        'cutoffs': torch.tensor(cutoffs, dtype=torch.int64, device=device),
        'conditions': None
    }

def add_info_metrics_all(
    model,
    items_path,
    output_path,
    gold_path=None,
    batch_size=64,
):
    """
    Adding general info metrics that don't care about gold trees.
    """
    items_path = Path(items_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    items_df = pd.read_csv(items_path)
    word_rows = expand_items_to_word_rows(items_df)
    num_rows = len(word_rows['Sentence'])


    with torch.no_grad():
        if gold_path is None: # Write all rows normally
            for metric in info_metrics_to_save:
                word_rows[metric] = []
            for start in range(0, num_rows, batch_size):
                end = min(start + batch_size, num_rows)
                batch_indices = range(start, end)
                batch = get_batch_from_word_rows(word_rows, start, end, device)
                dist_before, _, before_sentences = model.forward(
                    sentences=[s.copy() for s in batch['sentences']],
                    lengths=batch['lengths'],
                    cutoffs=batch['cutoffs']-1,
                )
                dist_after, _, after_sentences = model.forward(
                    sentences=[s.copy() for s in batch['sentences']],
                    lengths=batch['lengths'],
                    cutoffs=batch['cutoffs'],
                )
                metrics = get_info_metrics(dist_before, dist_after)

                before_sentences = [' '.join(sentence) for sentence in before_sentences]
                after_sentences = [' '.join(sentence) for sentence in after_sentences]
                for j, row_idx in enumerate(batch_indices):
                    word_rows['BeforeSentence'][row_idx] = before_sentences[j]
                    word_rows['AfterSentence'][row_idx] = after_sentences[j]

                for metric in info_metrics_to_save:
                    values = metrics[metric]
                    for j, row_idx in enumerate(batch_indices):
                        word_rows[metric].append(values[j])
        else: # Write rows with a gold parse differently
            df = pd.DataFrame(word_rows) 
            load_gold_trees(df, gold_path)
            amb_indices = df.index[df['ambiguity'] == 'ambiguous'].tolist()
            unamb_indices = df.index[df['ambiguity'] == 'unambiguous'].tolist()
            word_rows = df.to_dict('list')
            # Assume indices do form a range
            for metric in info_metrics_to_save:
                word_rows[metric] = []
                word_rows[f'{metric}_gold'] = []
            # Get metrics for ambiguous
            for start in range(amb_indices[0], amb_indices[-1], batch_size):
                end = min(start + batch_size, amb_indices[-1]+1)
                batch_indices = range(start, end)
                batch = get_batch_from_word_rows(word_rows, start, end, device)
                dist_before, _, before_sentences = model.forward(
                    sentences=[s.copy() for s in batch['sentences']],
                    lengths=batch['lengths'],
                    cutoffs=batch['cutoffs']-1,
                )
                dist_after, _, after_sentences = model.forward(
                    sentences=[s.copy() for s in batch['sentences']],
                    lengths=batch['lengths'],
                    cutoffs=batch['cutoffs'],
                )
                dist_gold = recovered_dist_like(
                    dist=dist_after, 
                    gold_trees=batch['gold_trees'],
                    cutoffs=batch['cutoffs']
                )
                metrics = get_info_metrics(dist_before, dist_after)
                gold_metrics = get_info_metrics(dist_before, dist_gold)
                gold_trees = dist_gold.argmax.detach().cpu().numpy()

                before_sentences = [' '.join(sentence) for sentence in before_sentences]
                after_sentences = [' '.join(sentence) for sentence in after_sentences]
                for j, row_idx in enumerate(batch_indices):
                    word_rows['BeforeSentence'][row_idx] = before_sentences[j]
                    word_rows['AfterSentence'][row_idx] = after_sentences[j]
                    word_rows['RecoveredTree'][row_idx] = gold_trees[j].tolist()

                for metric in info_metrics_to_save:
                    values = metrics[metric]
                    gold_values = gold_metrics[metric]
                    for j, row_idx in enumerate(batch_indices):
                        word_rows[metric].append(values[j])
                        word_rows[f'{metric}_gold'].append(gold_values[j])
            # Get metrics for unambiguous
            for start in range(unamb_indices[0], unamb_indices[-1], batch_size):
                end = min(start + batch_size, unamb_indices[-1]+1)
                batch_indices = range(start, end)
                batch = get_batch_from_word_rows(word_rows, start, end, device)
                dist_before, _, before_sentences = model.forward(
                    sentences=[s.copy() for s in batch['sentences']],
                    lengths=batch['lengths'],
                    cutoffs=batch['cutoffs']-1,
                )
                dist_after, _, after_sentences = model.forward(
                    sentences=[s.copy() for s in batch['sentences']],
                    lengths=batch['lengths'],
                    cutoffs=batch['cutoffs'],
                )
                metrics = get_info_metrics(dist_before, dist_after)

                before_sentences = [' '.join(sentence) for sentence in before_sentences]
                after_sentences = [' '.join(sentence) for sentence in after_sentences]
                for j, row_idx in enumerate(batch_indices):
                    word_rows['BeforeSentence'][row_idx] = before_sentences[j]
                    word_rows['AfterSentence'][row_idx] = after_sentences[j]

                for metric in info_metrics_to_save:
                    values = metrics[metric]
                    for j, row_idx in enumerate(batch_indices):
                        word_rows[metric].append(values[j])
                        word_rows[f'{metric}_gold'].append(np.nan)





    df = pd.DataFrame(word_rows)
    combine_multi_words(df, gold_path is not None)
    interleaved = []
    for m in info_metrics_to_save:
        interleaved.append(m)
        gold = f"{m}_gold"
        if gold in df.columns:
            interleaved.append(gold)
    front = ['EachWord'] + interleaved
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(output_path)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-v', '--version', type=int, required=True)
    parser.add_argument('-i', '--input_csv', default='data/phenomena/SAP/items_filler.csv')
    parser.add_argument('-o', '--output_csv', default=None)
    parser.add_argument('-g', '--gold_trees', default=None)
    parser.add_argument('--ckpt', default='last', choices=['val', 'mask', 'last'])
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
    model = Parser.load_from_checkpoint(best_ckpt)

    if args.output_csv is None:
        output_csv = f'out/test.csv'
    else:
        output_csv = args.output_csv

    df = add_info_metrics_all(
        model=model,
        items_path=args.input_csv,
        output_path=output_csv,
        gold_path=args.gold_trees,
        batch_size=args.batch_size,
    )

    print(f'Wrote metrics to {output_csv}')




        


  
