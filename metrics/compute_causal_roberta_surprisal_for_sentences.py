import argparse
import math
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, RobertaTokenizerFast

def compute_word_surprisals_for_sentence(sentence, tokenizer, model, device):
    """Accumulate token surprisals into word-by-word surprisal"""
    words = sentence.split(' ')
    inputs = tokenizer(
        words, 
        is_split_into_words=True,
        return_tensors='pt',
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get log probs
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # (seq_len, vocab_size)
    input_ids = inputs['input_ids'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    log_probs = torch.log_softmax(logits[:-1], dim=-1)
    
    # Specifically get surprisals of true tokens
    target_ids = input_ids[1:]
    token_surprisals = -log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1) # (seq_len - 1)
    token_surprisals = token_surprisals / math.log(2)
    token_surprisals = token_surprisals.detach().cpu().numpy()
    
    # Accumulate word surprisal by summing over token surprisal
    bpe_start = 'Ġ'
    word_rows = []
    current_word = ''
    current_word_tokens = []
    current_surprisal = 0
    for i, (token, surp) in enumerate(zip(tokens[1:], token_surprisals)):
        if token in ['</s>', '<s>', '<pad>']:
            continue

        if token.startswith(bpe_start):
            # Save current word
            if current_word_tokens:
                word_rows.append({
                    'EachWord': current_word,
                    'WordTokens': current_word_tokens,
                    'word_surprisal': current_surprisal
                })
            # Start new word
            clean_token = token.replace(bpe_start, '')
            current_word = clean_token
            current_word_tokens = [token]
            current_surprisal = surp
        else:
            # Continue current word 
            current_word += token
            current_word_tokens.append(token)
            current_surprisal += surp
                
    # Save final word
    if current_word_tokens:
        word_rows.append({
            'EachWord': current_word,
            'WordTokens': current_word_tokens,
            'word_surprisal': current_surprisal
        })
        
    return word_rows


def read_sentences(input_path):
    """Read one sentence per line from a plain-text file, skipping blank lines."""
    with open(input_path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_txt', required=True, help='Path to a .txt file with one sentence per line')
    parser.add_argument('-o', '--output_csv', default='out/causal_roberta_word_surprisals.csv')
    parser.add_argument('-m', '--model_name_or_path', default='../lightning_logs/causal_roberta/checkpoint-1200')
    args = parser.parse_args()

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = RobertaTokenizerFast.from_pretrained(
        args.model_name_or_path,
        add_prefix_space=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        local_files_only=True,
    )
    model.to(device) # type: ignore
    model.eval()

    sentences = read_sentences(args.input_txt)

    out = {
        'Sentence': [],
        'SentenceIndex': [],
        'word_pos': [],
        'EachWord': [],
        'WordTokens': [],
        'word_surprisal': [],
    }

    # Get word surprisals for each sentence
    for idx, sentence in enumerate(sentences):
        word_rows = compute_word_surprisals_for_sentence(
            sentence, 
            tokenizer, 
            model, 
            device
        )
        
        for pos, word_info in enumerate(word_rows, 1):
            out['Sentence'].append(sentence)
            out['SentenceIndex'].append(idx)
            out['word_pos'].append(pos)
            out['EachWord'].append(word_info['EachWord'])
            out['WordTokens'].append(word_info['WordTokens'])
            out['word_surprisal'].append(word_info['word_surprisal'])

    df = pd.DataFrame(out)
    df.to_csv(args.output_csv, index=False)
    print(f'Wrote word surprisals to {args.output_csv}')