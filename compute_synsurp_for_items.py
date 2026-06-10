import argparse
import math
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from model.synsurp_baseline import SynSurpRoBERTa


def compute_word_surprisals_for_sentence(sentence, model, device, kl=False):
    """Accumulate token surprisals into word-by-word surprisal"""
    words = sentence.split(' ')
    batch = {
        'sentences': [words],
        'tags': None
    }
    with torch.no_grad():
        out = model(batch)
        
    # Get preds for the first (and only) item in the batch
    word_labels = out['word_labels'][0]      # (word_len)
    lm_preds = out['lm_preds'][0]            # (word_len, num_words)
    supertag_preds = out['supertag_preds'][0] # (word_len, num_tags)
    word_probs = F.log_softmax(lm_preds, dim=-1) # includes bos
    
    # Calculate word surprisals directly.
    surprisals = -word_probs.gather(1, word_labels.unsqueeze(1)).squeeze(1)
    
    num_words = model.num_words
    num_tags = model.num_tags
    word_rows = []
    
    for i, word in enumerate(tqdm(words, desc='Words', leave=False)): 
        if i+1 < supertag_preds.shape[0]:
            # P(c_t+1 | w_t+1^*) 
            next_tag_given_true_word = F.log_softmax(supertag_preds[i+1, :], dim=-1) # (num_tags)
            # P(w_t+1|w_t)
            next_word_given_curr_word = word_probs[i] # (num_words)
        
            # P(c_t+1 | w_t+1)
            next_tag_given_next_word = torch.zeros((num_words, num_tags), device=device)
            batch_size = 512
            with torch.no_grad():
                for start_idx in range(0, num_words, batch_size):
                    end_idx = min(start_idx + batch_size, num_words)
                    batched_dummy_words = [
                        words[:i] + [model.id2word[w_id]] for w_id in range(start_idx, end_idx)
                    ]
                    dummy_batch = {
                        'sentences': batched_dummy_words,
                        'tags': None,
                    }
                    dummy_out = model(dummy_batch)
                    
                    next_tag_given_next_word[start_idx:end_idx] = F.log_softmax(
                        dummy_out['supertag_preds'][:, i+1, :], dim=-1
                    )
            
            # P(c_t+1|w_t) = sum(P(c_t+1|w_t+1) * P(w_t+1 | w_t))
            next_tag_given_curr_word = torch.logsumexp(
                next_tag_given_next_word + next_word_given_curr_word.unsqueeze(1), # w X t + 
                dim=0
            )
        
            if kl:
                # Compute KL(P(c_t+1|w_t+1) || P(c_t+1|w_t)) = KL (True || Predicted)
                synsurp = F.kl_div(
                    next_tag_given_curr_word,
                    next_tag_given_true_word,
                    reduction='sum',
                    log_target=True
                ).item()
            else: 
                # Compute synsurp = -log(sum_c(P(c_t+1|w_t+1*)P(c_t+1|w_t))
                synsurp = -torch.logsumexp(
                    next_tag_given_true_word + next_tag_given_curr_word,
                    dim=0
                ).item()
        else:
            synsurp = None

        metric_col = 'kl' if kl else 'syn_surp'
        word_rows.append({
            'EachWord': word,
            'word_surprisal': surprisals[i].item(),
            metric_col: synsurp
        })
        
    return word_rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-v', '--version', default=0)
    parser.add_argument('-i', '--input_csv', required=True)
    parser.add_argument('-o', '--output_csv', required=True)
    parser.add_argument('-kl', '--get_kl', action='store_true')
    parser.add_argument('--ckpt', default='val', choices=['val', 'last'])
    args = parser.parse_args()

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_dir = (
        Path('lightning_logs')
        / args.name
        / f'version_{args.version}'
        / 'checkpoints'
    )
    
    if args.ckpt == 'val':
        best_ckpt = next(ckpt_dir.glob('best_val_*.ckpt'))
    elif args.ckpt == 'last':
        best_ckpt = ckpt_dir / 'last.ckpt'
    else:
        best_ckpt = ckpt_dir / args.ckpt

    print(f'Loading checkpoint from {best_ckpt}')
    model = SynSurpRoBERTa.load_from_checkpoint(best_ckpt)
    model.to(device)
    model.eval()

    items_df = pd.read_csv(args.input_csv)
    base_columns = list(items_df.columns)
    metric_col = 'kl' if args.get_kl else 'syn_surp'
    final_columns = base_columns + ['word_pos', 'EachWord', 'word_surprisal', metric_col, 'SentenceIndex']
    pd.DataFrame(columns=final_columns).to_csv(output_path, index=False)

    # Get word surprisals for each sentence
    for idx, row in tqdm(items_df.iterrows(), total=len(items_df), desc='Processing sentences'):
        sentence = str(row['Sentence'])
        word_rows = compute_word_surprisals_for_sentence(
            sentence, model, device, kl=args.get_kl
        )
        
        sentence_out = {col: [] for col in final_columns}
        
        for pos, word_info in enumerate(word_rows, 1):
            for col in base_columns:
                sentence_out[col].append(row[col])
            
            sentence_out['word_pos'].append(pos)
            sentence_out['EachWord'].append(word_info['EachWord'])
            sentence_out['word_surprisal'].append(word_info['word_surprisal'])
            sentence_out[metric_col].append(word_info[metric_col])
            sentence_out['SentenceIndex'].append(idx)

        # Append to CSV incrementally
        df = pd.DataFrame(sentence_out)
        df.to_csv(output_path, mode='a', header=False, index=False)

    print(f'Finished writing word surprisals to {output_path}')


