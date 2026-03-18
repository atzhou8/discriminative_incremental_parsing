import pandas as pd
import stanza

from model.parser import Parser
from model.utils import get_info_metrics, uniform_dist_like, recovered_dist_like

info_metrics_to_save = [
    'kl_forward',
    'kl_root_shifted',
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
    for token in doc.sentences[0].tokens:
        # i.e. 'don't': ["do", "n't"]
        word_rows.append(
            {'unsplit': token.text,
            'split': [word.text for word in token.words]}
        )
    return word_rows
    
def expand_items_to_word_rows(item_df,sentence_gold_indices=None):
    base_columns = list(items_df.columns)
    new_columns = [
        'SentenceTokenized',
        'WordPosition', 
        'EachWord',
        'WordTokens', 
        'IsMultiWord', 
        'SentenceIndex', 
        'GoldTreeIndex'
    ]
    output_dict = {column: [] for column in base_columns + new_columns}

    for idx, (_, row) in enumerate(items_df.iterrows()):
        sentence_word_rows = create_word_rows_for_sentence(row['Sentence'])
        tokenized_sentence = [word_row['split'] for word_row in sentence_word_rows]
        for pos, word_row in enumerate(sentence_word_rows, start=1):
            each_word = word_row['unsplit']
            word_tokens = word_row['split']
            for token in word_tokens:
                for column in base_columns:
                    output_dict[column].append(row[column])

                output_dict['SentenceTokenized'].append(tokenized_sentence)
                output_dict['WordPosition'].append(int(pos))
                output_dict['EachWord'].append(each_word)
                output_dict['WordTokens'].append(word_tokens)
                output_dict['IsMultiWord'].apppend(len(word_tokens)>1)
                output_dict['SentenceIndex'].append(idx)
                output_dict['GoldTreeIndex'].append(
                    np.nan if sentence_gold_indices is None else senttence_gold_indices[idx]
                )
                output_dict['BeforeSentence'].append(np.nan)
                output_dict['AfterSentence'].append(np.nan)

    
    return output_dict

def get_batch_from_word_rows(word_rows, id_start, id_end, device):
    sentences = [word_rows['SentenceTokenized'][i] for i in range(id_start, id_end)]
    lengths = [len(sentence)+1 for sentence in sentences]
    cutoffs = [int(word_rows['WordPosition'])[i] for i in range(id_start, id_end)]

    return {
        'sentences': list(sentences), 
        'gold_trees': None, 
        'lengths': torch.tensor(lengths, dtype=torch.int64, device=device),
        'cutoffs': torch.tensor(cutoffs, dtype=torch.int64, device=device),
        'conditions': None
    }

def add_info_metrics_with_gold(
    model,
    items_path,
    output_path,
    gold_conllu_path='data/phenomena/SAP/amb_gold_length.conllu'
):
    """
    Specific code for computing evaluation metrics that require a gold tree
    """
    items_path = Path(items_path)
    output_csv_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    items_df = pd.read_csv(items_path)

def add_info_metrics_all(
    model,
    items_path,
    output_path,
    batch_size=64
):
    """
    Adding general info metrics that don't care about gold trees.
    """
    items_path = Path(items_path)
    output_csv_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)

    items_df = pd.read_csv(items_path)
    output_dict = expand_items_to_word_rows(items_df)
    num_rows = len(output_dict['Sentence'])

    with torch.no_grad():
        for start in range(0, num_rows, batch_size):
            end = min(start + batch_size, num_rows)
            batch = get_batch_from_word_rows(word_rows, stasrt, end, device)
            dist_before, _ before_sentences = model.forward(
                sentences=[s.copy() for s in batch['sentences']],
                lengths=batch['lengths'],
                cutoffs=batch['cutoffs']-1,
            )
            dist_after, _ before_after = model.forward(
                sentences=[s.copy() for s in batch['sentences']],
                lengths=batch['lengths'],
                cutoffs=batch['cutoffs'],
            )
            metrics = get_info_metrics(dist_before, dist_after)

            before_sentences = [' '.join(sentence) for sentence in before_sentences]
            after_sentences = [' '.join(sentence) for sentence in after_sentences]
            for j, row_idx in enumerate(row_indices):
                output_dict['before_sentence'][row_idx] = before_sentences[j]
                output_dict['after_sentence'][row_idx] = after_sentences[j]

            for metric_name in info_metrics_to_save:
                values = metrics[metric_name]
                for j, row_idx in enumerate(row_indices):
                    output_dict[metric_name][row_idx] = values[j]
    
    return pd.DataFrame(output_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-v', '--version', type=int, required=True)
    parser.add_argument('-i', '--input_csv', default='data/phenomena/SAP/items_filler.csv')
    parser.add_argument('-o', '--output_csv', default=None)
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
        batch_size=args.batch_size,
    )

    print(f'Wrote metrics to {output_csv}')




            
