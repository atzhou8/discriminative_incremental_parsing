import torch
import argparse
import pandas as pd
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm
from supar.structs import MatrixTree
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.parser import Parser
from metrics.utils import expand_sentences_to_word_rows, split_sentence_by_token, get_prefix_prompts_by_token, combine_multi_words
from models.info_metrics import get_info_metrics

def combine_marginals(marginals):
    """Concat all crfs into one big batch and mean across them"""
    max_len = max([m.shape[1] for m in marginals])
    padded_marginals = []
    for marg in marginals:
        to_pad = max_len - marg.shape[1]
        padded_marginals.append(F.pad(marg, (0, to_pad, 0, to_pad, 0, 0), value=0))

    all_marginals = torch.cat(padded_marginals, dim=0)
    return torch.mean(all_marginals, dim=0)


def anchorify_marginals(marginals, index):
    length = index + 2 # include anchor and root
    batch_size = marginals.shape[0]
    new_margs = torch.zeros(length, length, device=marginals.device) # +2 for anchor and root

    # ingoing anchor scores
    future_parent_scores = marginals[:, index+1:]
    anchor_parent_scores = future_parent_scores.sum(dim=1)

    # outgoing anchor scores
    # future_child_scores = marginals[index+1:, :]
    # anchor_child_scores = future_child_scores.sum(dim=0)
    anchor_child_scores = torch.zeros(1, length)
    anchor_child_scores[:, 0] = 1

    # fill in scores
    new_margs[:index+1, :index+1] = marginals[:index+1, :index+1] # copy valid node scores
    if index+1 < marginals.shape[1]: # if future nodes exist
        new_margs[:, index+1] = anchor_parent_scores[:length]
    new_margs[-1, :] = anchor_child_scores 
    # new_crf = MatrixTree(
    #     scores=new_scores,
    #     lens=length*torch.ones_like(crf.lens),
    #     multiroot=True
    # ) for debugging individual trees

    return new_margs

def print_trees(crf, sentences):
    trees = crf.argmax
    for i, sent in enumerate(sentences):
        tree = trees[i, 1:]
        print('-'*80)
        for i, o in enumerate(zip(tree, sent)):
            s, n = o
            print(i+1, s, n)

def print_prefix(crf, sentences, i):
    anchor_sentences = [s[:i]+['<anchor>'] for s in sentences]
    print_trees(crf, anchor_sentences)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INFO_METRICS_TO_SAVE = [
    'kl_divergence',
]


if __name__ == '__main__':
    c=3
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='transfer')
    parser.add_argument('-v', '--version', type=int, default=0)
    parser.add_argument('-i', '--data_csv', default=f'/home/jhu/azhou23/scratch_jhale1/discriminative_incremental_parsing/data/phenomena/MECO/my_data/passages/English/{c}.txt')
    parser.add_argument('-c', '--continuation_csv', default=f'/home/jhu/azhou23/scratch_jhale1/discriminative_incremental_parsing/continuations/MECO/English/{c}_continuations.csv')
    parser.add_argument('-o', '--output_csv', default='out.csv')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    args = parser.parse_args()

    ckpt_dir = (
        Path('./lightning_logs')
        / args.name
        / f'version_{args.version}'
        / 'checkpoints'
    )
    best_ckpt = next(ckpt_dir.glob(f'best_val_epoch=*.ckpt'))
    parser = Parser.load_from_checkpoint(best_ckpt, map_location=device)
    parser.to(device)

    data = pd.read_csv(args.data_csv, sep='\t', header=None, names=['Sentence'])
    cont = pd.read_csv(args.continuation_csv)
    sentences = data['Sentence'].to_list()
    word_rows = expand_sentences_to_word_rows(data)
    batch_size = args.batch_size

    i = 0 
    for metric in INFO_METRICS_TO_SAVE:
        word_rows[metric] = []
        for sentence in tqdm(sentences, desc='Sentence'):
            conts_for_sentence = cont[cont['orig_sentence']==sentence]
            prefixes = conts_for_sentence['prefix_position'].unique()
            prev_marginals = None
            for prefix in tqdm(prefixes, leave=False, desc='Prefix'):
                conts_for_prefix = conts_for_sentence[conts_for_sentence['prefix_position']==prefix]
                conts_for_prefix = conts_for_prefix['continuation'].to_list()
                marginals_for_prefix = []
                # for b in tqdm(range(2), leave=False, desc='Batch'):
                for b in tqdm(range(len(conts_for_prefix)//args.batch_size), leave=False, desc='Batch'):
                    cont_batch = conts_for_prefix[b*args.batch_size:(b+1)*args.batch_size]
                    cont_batch = [split_sentence_by_token(s) for s in cont_batch]
                    length_batch = torch.tensor(
                        [len(cont)+1 for cont in cont_batch],
                        dtype=torch.long,
                        device=parser.device
                    )
                    with torch.no_grad():
                        marginals_for_prefix.append(parser.forward(cont_batch, length_batch)['crf'].marginals)
                curr_marginals = combine_marginals(marginals_for_prefix)
                if prev_marginals is None:
                    kl = 0
                else:
                    dist_before = anchorify_marginals(prev_marginals, prefix+1)
                    dist_after = anchorify_marginals(curr_marginals, prefix+1)
                    kl = F.kl_div(
                        target=dist_after+1e-10,
                        input=torch.log(dist_before+1e-10),
                        reduction='sum',
                        log_target=False, 
                    ).item()
                word_rows['kl_divergence'].append(kl)
                i = i + 1
                prev_marginals = curr_marginals
            

    word_rows = pd.DataFrame(word_rows)
    combine_multi_words(word_rows, INFO_METRICS_TO_SAVE)
    word_rows.to_csv(args.output_csv)


