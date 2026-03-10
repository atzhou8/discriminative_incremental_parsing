import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from urllib.parse import unquote

from model.parser import Parser
from model.utils import get_info_metrics

torch.set_float32_matmul_precision('medium')


def _get_info_metric_names():
    names = [
        'entropy_before',
        'entropy_after',
        'entropy_reduction',
        'entropy_change',
        'cross_entropy_forward',
        'cross_entropy_backward',
        'kl_forward',
        'kl_backward',
        'kl_symmetric',
        'js_geo',
    ]
    for alpha in [2, 3, 5]:
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


def _expand_items_to_word_rows_dict(
    items_df,
):
    base_columns = list(items_df.columns)
    output_columns = base_columns + ['WordPosition', 'EachWord']
    output_dict = {column: [] for column in output_columns}

    for _, row in items_df.iterrows():
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

    return output_dict


def add_info_metrics_to_fillers_csv(
    model,
    fillers_csv_path,
    output_csv_path,
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
	


    output_dict = _expand_items_to_word_rows_dict(
        items_df,
    )

    num_rows = len(output_dict['Sentence'])
    output_dict['before_sentence'] = [np.nan] * num_rows
    output_dict['after_sentence'] = [np.nan] * num_rows
    for metric_name in metric_names:
        output_dict[metric_name] = [np.nan] * num_rows

    valid_positions = []
    valid_sentences = []
    valid_lengths = []
    for i in range(num_rows):
        sentence = _normalize_sentence(output_dict['Sentence'][i])
        words = sentence.split()
        cutoff = int(output_dict['WordPosition'][i])
        valid_positions.append(i)
        valid_sentences.append(words)
        valid_lengths.append(len(words) + 1)

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

    pd.DataFrame(output_dict).to_csv(output_csv_path, index=False)


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
        output_csv = f'out/{args.name}_fillers_info_metrics.csv'
    else:
        output_csv = args.output_csv

    add_info_metrics_to_fillers_csv(
        model=model,
        fillers_csv_path=args.input_csv,
        output_csv_path=output_csv,
        batch_size=args.batch_size,
    )

    print(f'Wrote metrics to {output_csv}')
