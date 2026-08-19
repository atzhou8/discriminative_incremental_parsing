import torch
import pandas as pd

from pathlib import Path
from supar.structs import MatrixTree
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.parser import Parser
from utils import expand_sentences_to_word_rows, split_sentence_by_word

lm_name = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(lm_name, add_prefix_space=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
lm = AutoModelForCausalLM.from_pretrained(lm_name)

parser_name = 'parser_full'
parser_version = 1

ckpt_dir = (
    Path('./lightning_logs')
    / parser_name
    / f'version_{parser_version}'
    / 'checkpoints'
)
best_ckpt = next(ckpt_dir.glob(f'best_val_epoch=*.ckpt'))
parser = Parser.load_from_checkpoint(best_ckpt)

def sample_continuations(
        sentence, 
        max_new_tokens,
        num_continuations,
        temperature=1
):
    words = split_sentence_by_word(sentence)
    prefixes = [words[:i+1] for i in range(len(words))]
    bad_words = ['\n', '\n\n']
    bad_words_ids = [tokenizer.encode(w) for w in bad_words]

    inputs = tokenizer(
        prefixes,
        is_split_into_words=True,
        return_tensors='pt',
        truncation=True,
        padding_side='left',
        padding=True,
    )
    continuations = lm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_continuations,
        temperature=temperature,
        tokenizer=tokenizer,
        bad_words_ids=bad_words_ids, # ignore \n
        forced_bos_token_id=tokenizer.encode(' '), # start with new word
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

    output = {' '.join(prefix): continuations[i*num_continuations:(i+1)*num_continuations] for i, prefix in enumerate(prefixes)}

    return output


def anchorify_edge_weights(crf, index):
    scores = crf.scores # (b, n, n)
    batch_size = scores.shape[0]
    length = index + 2 # include anchor and root
    new_scores = torch.zeros(batch_size, length, length) # +2 for anchor and root

    # ingoing anchor scores
    ingoing_scores = scores[:, :, index+1:]
    ingoing_scores = torch.logsumexp(ingoing_scores, dim=-1, keepdim=True)

    # outgoing anchor scores
    # outgoing_scores = scores[:, index+1:, :]
    # outgoing_scores = reduce(outgoing_scores, 'b n m -> b 1 m', 'sum')
    # outgoing_scorse = outgoing_scores[:, :, :length]
    outgoing_scores = torch.zeros(batch_size, 1, length)
    outgoing_scores[:, :, 0] = 1

    # fill in scores
    new_scores[:, :index+1, :index+1] = scores[:, :index+1, :index+1] # copy valid node scores
    new_scores[:, :, index+1:] = ingoing_scores[:, :length, :]
    new_scores[:, index+1:, :] = outgoing_scores

    new_crf = MatrixTree(
        scores=new_scores,
        lens=length*torch.ones_like(crf.lens),
        multiroot=True
    )
    return new_crf


input_csv = Path('/home/azhou23/scratchjhale1/discriminative_incremental_parsing/data/phenomena/MECO/my_data/passages/English/1.txt')
data = pd.read_csv(input_csv, sep='\t', header=None, names=['Sentence'])
sentences = data['Sentence'].to_list()
word_rows = expand_sentences_to_word_rows(sentences)
for sentence in sentences:
    sentence_continuations = sample_continuations(
        sentence,
        max_new_tokens=100,
        num_continuations=128,
    )
    for i, continuations in sentence_continuations.items():
        lengths = torch.tensor(
            [len(cont) for cont in continuations],
            dtype=torch.long
            device=parser.device
        )
        out = parser.forward(continuations, lengths)
    



# k = sample_continuations('The girl fed the lamb', max_new_tokens=100, num_continuations=4)

# sentences = [
#     'The boy found the chicken stayed surprisingly happy in the new barn.',
#     'Although the boy attacked the chicken stayed surprisingly happy as if nothing happened.',
#     'The boy fed the chicken stayed surprisingly happy despite having a mild allergic reaction.' 
# ]
# index=3
# split_sentences = [split_for_parser(s) for s in sentences]
# anchor_sentences = [s[:index]+['<anchor>'] for s in split_sentences]
# lengths = torch.tensor(
#     [len(s)+1 for s in split_sentences], dtype=torch.long
# )
# lengths.to(parser.device) 
# out = parser.forward(split_sentences, lengths)
# crf = out['crf']
# anchor_trees = anchorify_edge_weights(crf, index)

# def print_trees(crf, sentences):
#     trees = crf.argmax
#     for i, sent in enumerate(sentences):
#         tree = trees[i, 1:]
#         print('-'*80)
#         for i, o in enumerate(zip(tree, sent)):
#             s, n = o
#             print(i+1, s, n)

# for i in range(8):
#     anchor_trees = anchorify_edge_weights(crf, i)
#     anchor_sentences = [s[:i]+['<anchor>'] for s in split_sentences]
#     print('*'*80)
#     print_trees(anchor_trees, anchor_sentences)