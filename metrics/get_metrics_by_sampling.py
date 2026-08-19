import torch
import argparse
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from supar.structs import MatrixTree
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.parser import Parser
from metrics.utils import expand_sentences_to_word_rows, split_sentence_by_token, combine_multi_words
from models.info_metrics import get_info_metrics

def sample_continuations(
        sentence, 
        max_new_tokens,
        num_continuations,
        temperature=1
):
    words = split_sentence_by_token(sentence)
    prefixes = [words[:i+1] for i in range(len(words)-1)] # don't generate from end of sentence

    inputs = tokenizer(
        prefixes,
        is_split_into_words=True,
        return_tensors='pt',
        truncation=True,
        padding_side='left',
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        continuations = lm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_continuations,
            temperature=temperature,
            tokenizer=tokenizer,
            bad_words_ids=BAD_WORDS_IDS, # ignore \n
            # forced_bos_token_id=tokenizer.encode(' '), # start with new word
            stop_strings=['.', '!', '?'], # end with sentence
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            num_beams=1,
            use_cache=True,
        )
    continuations = tokenizer.batch_decode(
        continuations, 
        skip_special_tokens=True
    )

    output = {i: continuations[i*num_continuations:(i+1)*num_continuations] 
              for i, prefix in enumerate(prefixes)}
    output[len(prefixes)] = [sentence for _ in range(num_continuations)]

    return output


def anchorify_crf(crf, index):
    scores = crf.scores # (b, n, n)
    batch_size = scores.shape[0]
    length = index + 2 # include anchor and root
    new_scores = -1e32 * torch.ones(batch_size, length, length, device=scores.device) # +2 for anchor and root

    # ingoing anchor scores
    ingoing_scores = scores[:, :, index+1:]
    ingoing_scores = torch.logsumexp(ingoing_scores, dim=-1, keepdim=True)

    # outgoing anchor scores
    outgoing_scores = -1e32 * torch.ones(batch_size, 1, length)
    outgoing_scores[:, :, 0] = 1

    # fill in scores
    new_scores[:, :index+1, :index+1] = scores[:, :index+1, :index+1] # copy valid node scores
    if index+1 < scores.shape[1]: # if future nodes exist
        new_scores[:, :, index+1:] = ingoing_scores[:, :length, :]
    new_scores[:, index+1:, :] = outgoing_scores

    mean_denom = torch.tensor([batch_size], dtype=torch.float32, device=scores.device)
    new_scores = torch.logsumexp(new_scores, dim=0, keepdim=True) - torch.log(mean_denom)

    new_crf = MatrixTree(
        scores=new_scores,
        lens=torch.tensor([length], dtype=torch.long, device=scores.device),
        multiroot=True
    )

    # new_crf = MatrixTree(
    #     scores=new_scores,
    #     lens=length*torch.ones_like(crf.lens),
    #     multiroot=True
    # ) for debugging individual trees

    return new_crf



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


device = torch.device('cuda')
lm_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(lm_name, add_prefix_space=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
lm = AutoModelForCausalLM.from_pretrained(lm_name)
lm.to(device)

BAD_WORDS_IDS = [
    [tok_id] for tok_id in range(len(tokenizer))
    if '\n' in tokenizer.decode([tok_id])
]

INFO_METRICS_TO_SAVE = [
    'kl_backward',
    'renyi_divergence_backward_2',
    'renyi_divergence_backward_3',
    'renyi_divergence_backward_4',
    'renyi_divergence_backward_5',
    'renyi_divergence_backward_6',  
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-v', '--version', type=int, required=True)
    parser.add_argument('-i', '--input_csv', default='data/phenomena/SAP/items_filler.csv')
    parser.add_argument('-o', '--output_csv', default=None)
    # parser.add_argument('--ckpt', default='val', choices=['val', 'cutoff', 'last'])
    # parser.add_argument('-m', '--model_type', default='parser', choices=['parser', 'tagger'])
    # parser.add_argument('--batch-size', type=int, default=64)
    # parser.add_argument('--input_is_csv', action='store_true')
    args = parser.parse_args()

    ckpt_dir = (
        Path('./lightning_logs')
        / args.name
        / f'version_{args.version}'
        / 'checkpoints'
    )
    best_ckpt = next(ckpt_dir.glob(f'best_val_epoch=*.ckpt'))
    parser = Parser.load_from_checkpoint(best_ckpt)
    parser.to(device)

    # input_csv = Path('/home/azhou23/scratchjhale1/discriminative_incremental_parsing/data/phenomena/MECO/my_data/passages/English/1.txt')
    data = pd.read_csv(args.input_csv, sep='\t', header=None, names=['Sentence'])
    sentences = data['Sentence'].to_list()
    word_rows = expand_sentences_to_word_rows(data)

    for metric in INFO_METRICS_TO_SAVE:
        word_rows[metric] = []

    for sentence in sentences:
        sentence_continuations = sample_continuations(
            sentence,
            max_new_tokens=200,
            num_continuations=128,
        )
        prev_crf = None
        prev_continuation = None
        for i, continuations in tqdm(sentence_continuations.items()):
            continuations = [split_sentence_by_token(s) for s in continuations]
            lengths = torch.tensor(
                [len(cont)+1 for cont in continuations],
                dtype=torch.long,
                device=parser.device
            )
            with torch.no_grad():
                curr_crf = parser.forward(continuations, lengths)['crf']
            if prev_crf == None:
                metrics = {metric: [0] for metric in INFO_METRICS_TO_SAVE}
            else:
                dist_before = anchorify_crf(prev_crf, i+1)
                dist_after = anchorify_crf(curr_crf, i+1)
                metrics = get_info_metrics(dist_before, dist_after)

            for metric in INFO_METRICS_TO_SAVE:
                value = metrics[metric]
                word_rows[metric].append(value[0])

            prev_crf = curr_crf
            prev_continuation = continuations

    word_rows = pd.DataFrame(word_rows)
    combine_multi_words(word_rows, INFO_METRICS_TO_SAVE)
    word_rows.to_csv(args.output_csv)
    print(f'Wrote to {args.output_csv}')

