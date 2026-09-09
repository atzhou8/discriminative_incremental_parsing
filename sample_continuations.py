import torch
import argparse
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from supar.structs import MatrixTree
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.parser import Parser
from metrics.utils import expand_sentences_to_word_rows, split_sentence_by_token, get_prefix_prompts_by_token, combine_multi_words
from models.info_metrics import get_info_metrics


def sample_from_BOS(
    max_new_tokens,
    num_continuations=32,
    temperature=1,
    batch_size=4,
):
    assert num_continuations % batch_size == 0

    inputs = tokenizer(
        [""],
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    samples = []
    for _ in tqdm(range(num_continuations // batch_size), leave=False, desc='Batch'):
        with torch.no_grad():
            out = lm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=batch_size,
                temperature=temperature,
                tokenizer=tokenizer,
                bad_words_ids=BAD_WORDS_IDS,
                stop_strings=[tokenizer.eos_token],
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.95,
                # repetition_penalty=1.3,
                # no_repeat_ngram_size=3,
                do_sample=True,
                num_beams=1,
                use_cache=True,
            )
    samples.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return samples

def sample_continuations(
        sentence,
        max_new_tokens,
        num_continuations=32,
        temperature=1,
        batch_size=4,
):
    """Given a full sentence, generate possible sentence continuations at every
    prefix"""
    assert num_continuations % batch_size == 0

    prefixes = get_prefix_prompts_by_token(sentence)  # [(words_so_far, prefix_text), ...]
    prefix_strings = [text for _, text in prefixes]
    inputs = tokenizer(
        prefix_strings,
        return_tensors='pt',
        truncation=True,
        padding_side='left',
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    all_continuations = {i: [] for i in range(len(prefixes))}

    for _ in tqdm(range(num_continuations // batch_size), leave=False, desc='Batch'):
        with torch.no_grad():
            out = lm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=batch_size,
                temperature=temperature,
                tokenizer=tokenizer,
                bad_words_ids=BAD_WORDS_IDS,
                stop_strings=[tokenizer.eos_token],
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.95,
                # repetition_penalty=1.3,
                # no_repeat_ngram_size=3,
                do_sample=True,
                num_beams=1,
                use_cache=True,
            )

        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

        for i, _ in enumerate(prefixes):
            batch = decoded[i*batch_size:(i+1)*batch_size]
            for j, cont in enumerate(batch):
                sents = sent_tokenize(cont)
                batch[j] = sents[0] if sents else cont
            all_continuations[i].extend(batch)

    all_continuations[len(prefixes)] = [sentence for _ in range(num_continuations)]
    return all_continuations


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lm_name = 'gpt2-large'
tokenizer = AutoTokenizer.from_pretrained(lm_name, add_bos_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'


lm = AutoModelForCausalLM.from_pretrained(lm_name, torch_dtype=torch.float16)
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
    parser.add_argument('-i', '--input_csv', default=None)
    parser.add_argument('-o', '--output_csv', default=None)
    parser.add_argument('-t', '--max_new_tokens', type=int, default=200)
    parser.add_argument('-c', '--num_continuations', type=int, default=8)
    args = parser.parse_args()


    data = pd.read_csv(args.input_csv, sep='\t', header=None, names=['Sentence'])
    sentences = data['Sentence'].to_list()

    continuations = {
        'orig_sentence': [],
        'prefix_position': [],
        'continuation': []
    }
    if args.input_csv is not None:
        for sentence in tqdm(sentences, desc='Sentence'):
            sentence_continuations = sample_continuations(
                sentence,
                max_new_tokens=args.max_new_tokens,
                num_continuations=args.num_continuations,
            )

            for prefix, prefix_continuations in sentence_continuations.items():
                for cont in prefix_continuations:
                    continuations['orig_sentence'].append(sentence)
                    continuations['prefix_position'].append(prefix)
                    continuations['continuation'].append(cont)
    else:
        bos_continuations = sample_from_BOS(
            max_new_tokens=args.max_new_tokens,
            num_continuations=args.num_continuations,
        )
        for cont in bos_continuations:
            continuations['orig_sentence'].append('')
            continuations['prefix_position'].append(0)
            continuations['continuation'].append(cont)

    continuations = pd.DataFrame(continuations)
    continuations.to_csv(args.output_csv)
    print(f'Wrote to {args.output_csv}')








    # ZZZ Old code for actually computing metrics
    # word_rows = expand_sentences_to_word_rows(data)

    # for metric in INFO_METRICS_TO_SAVE:
    #     word_rows[metric] = []
    #     prev_crf = None
    #     prev_continuation = None
    #     for i, continuations in tqdm(sentence_continuations.items(), leave=False, desc='prefixes'):
    #         continuations = [split_sentence_by_token(s) for s in continuations]
    #         lengths = torch.tensor(
    #             [len(cont)+1 for cont in continuations],
    #             dtype=torch.long,
    #             device=parser.device
    #         )
    #         with torch.no_grad():
    #             curr_crf = parser.forward(continuations, lengths)['crf']
    #         if prev_crf == None:
    #             metrics = {metric: [0] for metric in INFO_METRICS_TO_SAVE}
    #         else:
    #             dist_before = anchorify_crf(prev_crf, i+1)
    #             dist_after = anchorify_crf(curr_crf, i+1)
    #             metrics = get_info_metrics(dist_before, dist_after)

    #         for metric in INFO_METRICS_TO_SAVE:
    #             value = metrics[metric]
    #             word_rows[metric].append(value[0])

    #         prev_crf = curr_crf
    #         prev_continuation = continuations

    # word_rows = pd.DataFrame(word_rows)
    # combine_multi_words(word_rows, INFO_METRICS_TO_SAVE)
    # word_rows.to_csv(args.output_csv)
    # print(f'Wrote to {args.output_csv}')

