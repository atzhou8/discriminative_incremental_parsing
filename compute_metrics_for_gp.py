import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from urllib.parse import unquote
from stanza.utils.conll import CoNLL

from model.parser import Parser
from model.utils import get_info_metrics, uniform_dist_like, recovered_dist_like

torch.set_float32_matmul_precision('medium')


def _get_info_metric_names():
    names = [
        # 'entropy_before',
        # 'entropy_after',
        # 'entropy_reduction',
        # 'entropy_change',
        'kl_forward',
        'kl_root_shifted',
        'kl_backward',
        'kl_symmetric',
        'js_geo',
        'cross_entropy_forward',
        'cross_entropy_backward',
    ]
    for alpha in [5]:
        names.extend(
            [
                f'entropy_before_renyi_{alpha}',
                f'entropy_after_renyi_{alpha}',
                f'renyi_divergence_forward_{alpha}',
                f'renyi_divergence_backward_{alpha}',
                f'renyi_divergence_symmetric_{alpha}',
            ]
        )
    return names


def _to_python_list(metric_value):
    metric_value = torch.as_tensor(metric_value).detach().cpu().numpy()
    return np.atleast_1d(metric_value).tolist()


def _normalize_sentence(value):
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return ''
    return unquote(text)


def _load_gold_trees_by_index(gold_conllu_path):
    doc = CoNLL.conll2doc(str(gold_conllu_path))
    gold_trees = []
    for sent in doc.sentences:
        gold_trees.append(np.asarray(
            [int(word.head) for word in sent.words],
            dtype=np.int64,
        ))
    return gold_trees


def _expand_items_to_word_rows_dict(
    items_df,
    sentence_gold_indices,
):
    base_columns = list(items_df.columns)
    output_columns = base_columns + ['WordPosition', 'EachWord', 'SentenceIndex', 'GoldTreeIndex']
    output_dict = {column: [] for column in output_columns}

    for sentence_idx, (_, row) in enumerate(items_df.iterrows()):
        normalized_sentence = _normalize_sentence(row['Sentence'])
        words = normalized_sentence.split()
        if len(words) == 0:
            continue

        for pos, each_word in enumerate(words, start=1):
            for column in base_columns:
                if column == 'Sentence':
                    output_dict[column].append(normalized_sentence)
                else:
                    output_dict[column].append(row[column])
            output_dict['WordPosition'].append(np.int64(pos))
            output_dict['EachWord'].append(each_word)
            output_dict['SentenceIndex'].append(np.int64(sentence_idx))
            gold_tree_idx = sentence_gold_indices[sentence_idx]
            output_dict['GoldTreeIndex'].append(
                np.nan if gold_tree_idx is None else np.int64(gold_tree_idx)
            )

    return output_dict


def add_info_metrics_to_fillers_csv(
    model,
    fillers_csv_path,
    output_csv_path,
    gold_conllu_path='data/phenomena/SAP/amb_gold_length.conllu',
    batch_size=64,
    device=None,
):
    metric_names = _get_info_metric_names()
    fillers_csv_path = Path(fillers_csv_path)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    items_df = pd.read_csv(fillers_csv_path, low_memory=False).reset_index(drop=True)
    gold_trees = _load_gold_trees_by_index(gold_conllu_path)

    ambiguous_mask = pd.Series([True] * len(items_df), index=items_df.index)
    if 'ambiguity' in items_df.columns:
        ambiguous_mask = items_df['ambiguity'].astype(str).str.lower() == 'ambiguous'
        unambiguous_mask = ~ambiguous_mask
        for column_name in ['recovered', 'recoverd', 'confused', 'gold', 'predicted']:
            if column_name in items_df.columns:
                items_df.loc[unambiguous_mask, column_name] = np.nan

    gold_name = Path(gold_conllu_path).name
    sentence_gold_indices = [None] * len(items_df)
    if (
        len(items_df) > len(gold_trees)
        and 'ambiguity' in items_df.columns
        and gold_name.startswith('amb_gold')
    ):
        ambiguous_indices = items_df.index[ambiguous_mask].tolist()
        if len(ambiguous_indices) != len(gold_trees):
            raise ValueError(
                f'Ambiguous rows ({len(ambiguous_indices)}) and gold trees ({len(gold_trees)}) do not align. '
                f'Input={fillers_csv_path}, gold={gold_conllu_path}'
            )
        for gold_idx, sentence_idx in enumerate(ambiguous_indices):
            sentence_gold_indices[sentence_idx] = gold_idx
    elif len(items_df) == len(gold_trees):
        sentence_gold_indices = list(range(len(items_df)))
    else:
        raise ValueError(
            f'Input rows ({len(items_df)}) and gold trees ({len(gold_trees)}) do not align. '
            f'Input={fillers_csv_path}, gold={gold_conllu_path}'
        )


    output_dict = _expand_items_to_word_rows_dict(
        items_df,
        sentence_gold_indices,
    )

    num_rows = len(output_dict['Sentence'])
    output_dict['before_sentence'] = [np.nan] * num_rows
    output_dict['after_sentence'] = [np.nan] * num_rows
    output_dict['gold_parse'] = [np.nan] * num_rows
    output_dict['recovered_parse'] = [np.nan] * num_rows
    for metric_name in metric_names:
        output_dict[metric_name] = [np.nan] * num_rows
        output_dict[f'{metric_name}_confused'] = [np.nan] * num_rows
        output_dict[f'{metric_name}_recovered'] = [np.nan] * num_rows

    valid_positions = []
    valid_sentences = []
    valid_lengths = []
    ambiguous_positions = []
    ambiguous_sentences = []
    ambiguous_lengths = []
    ambiguous_gold_trees = []
    for i in range(num_rows):
        sentence = _normalize_sentence(output_dict['Sentence'][i])
        words = sentence.split()
        valid_positions.append(i)
        valid_sentences.append(words)
        valid_lengths.append(len(words) + 1)
        gold_tree_idx = output_dict['GoldTreeIndex'][i]
        if pd.notna(gold_tree_idx):
            ambiguous_positions.append(i)
            ambiguous_sentences.append(words)
            ambiguous_lengths.append(len(words) + 1)
            ambiguous_gold_trees.append(gold_trees[int(gold_tree_idx)])

    with torch.no_grad():
        for start in range(0, len(valid_sentences), batch_size):
            end = min(start + batch_size, len(valid_sentences))
            row_indices = valid_positions[start:end]
            batch_sentences = valid_sentences[start:end]
            batch_lengths = torch.tensor(
                valid_lengths[start:end],
                dtype=torch.int64,
                device=device,
            )
            batch_cutoffs = torch.tensor(
                [
                    int(output_dict['WordPosition'][idx])
                    for idx in row_indices
                ],
                dtype=torch.int64,
                device=device,
            )

            dist_before, _, before_cut_sentences = model.forward(
                [s.copy() for s in batch_sentences],
                batch_lengths,
                cutoffs=batch_cutoffs - 1,
            )
            dist_after, _, after_cut_sentences = model.forward(
                [s.copy() for s in batch_sentences],
                batch_lengths,
                cutoffs=batch_cutoffs,
            )
            metrics = get_info_metrics(dist_before, dist_after)

            before_strings = [' '.join(sentence) for sentence in before_cut_sentences]
            after_strings = [' '.join(sentence) for sentence in after_cut_sentences]

            for j, row_idx in enumerate(row_indices):
                output_dict['before_sentence'][row_idx] = before_strings[j]
                output_dict['after_sentence'][row_idx] = after_strings[j]

            for metric_name in metric_names:
                values = _to_python_list(metrics[metric_name])
                for j, row_idx in enumerate(row_indices):
                    output_dict[metric_name][row_idx] = values[j]

        for start in range(0, len(ambiguous_sentences), batch_size):
            end = min(start + batch_size, len(ambiguous_sentences))
            row_indices = ambiguous_positions[start:end]
            batch_sentences = ambiguous_sentences[start:end]
            batch_lengths = torch.tensor(
                ambiguous_lengths[start:end],
                dtype=torch.int64,
                device=device,
            )
            batch_cutoffs = torch.tensor(
                [
                    int(output_dict['WordPosition'][idx])
                    for idx in row_indices
                ],
                dtype=torch.int64,
                device=device,
            )
            batch_gold_trees = np.asarray(ambiguous_gold_trees[start:end], dtype=object)

            dist_before, _, _ = model.forward(
                [s.copy() for s in batch_sentences],
                batch_lengths,
                cutoffs=batch_cutoffs - 1,
            )
            dist_after, _, _ = model.forward(
                [s.copy() for s in batch_sentences],
                batch_lengths,
                cutoffs=batch_cutoffs,
            )
            dist_after_confused = uniform_dist_like(dist_after)
            dist_after_recovered = recovered_dist_like(dist_after, batch_gold_trees, batch_cutoffs)

            metrics_confused = get_info_metrics(dist_before, dist_after_confused)
            metrics_recovered = get_info_metrics(dist_before, dist_after_recovered)
            recovered_argmax = dist_after_recovered.argmax.detach().cpu().tolist()

            for j, row_idx in enumerate(row_indices):
                token_count = len(batch_sentences[j])
                output_dict['gold_parse'][row_idx] = batch_gold_trees[j].tolist()[:token_count]
                output_dict['recovered_parse'][row_idx] = recovered_argmax[j][1:1 + token_count]

            for metric_name in metric_names:
                values_confused = _to_python_list(metrics_confused[metric_name])
                values_recovered = _to_python_list(metrics_recovered[metric_name])
                for j, row_idx in enumerate(row_indices):
                    output_dict[f'{metric_name}_confused'][row_idx] = values_confused[j]
                    output_dict[f'{metric_name}_recovered'][row_idx] = values_recovered[j]

    output_df = pd.DataFrame(output_dict)
    if 'GoldTreeIndex' in output_df.columns:
        output_df = output_df.drop(columns=['GoldTreeIndex'])
    trailing_columns = [
        column for column in ['gold_parse', 'recovered_parse', 'before_sentence', 'after_sentence']
        if column in output_df.columns
    ]
    leading_columns = [column for column in output_df.columns if column not in trailing_columns]
    output_df = output_df[leading_columns + trailing_columns]
    output_df.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-v', '--version', type=int, required=True)
    parser.add_argument('-i', '--input_csv', default='data/phenomena/SAP/items_filler.csv')
    parser.add_argument('-o', '--output_csv', default=None)
    parser.add_argument('--ckpt', default='last', choices=['val', 'mask', 'last'])
    parser.add_argument('--gold_conllu_path', default='data/phenomena/SAP/amb_gold_length.conllu')
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
        output_csv = f'out/{args.name}_fillers_info_metrics.csv'
    else:
        output_csv = args.output_csv

    add_info_metrics_to_fillers_csv(
        model=model,
        fillers_csv_path=args.input_csv,
        output_csv_path=output_csv,
        gold_conllu_path=args.gold_conllu_path,
        batch_size=args.batch_size,
    )

    print(f'Wrote metrics to {output_csv}')
