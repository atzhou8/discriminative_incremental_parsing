import argparse
import pandas as pd
import numpy as np

from compute_metrics_for_items import INFO_METRICS_TO_SAVE

def order_columns(df, additional_columns=None):
    if additional_columns is None:
        additional_columns = []

    metrics = []
    for metric in INFO_METRICS_TO_SAVE:
        metrics.append(f'{metric}_s')
        metrics.append(metric)
        for additional_col in additional_columns:
            metrics.append(f'{metric}_{additional_col}_s')
            metrics.append(f'{metric}_{additional_col}')
    front = ['EachWord', 'IsAdjunct'] + metrics
    df = df[front + [c for c in df.columns if c not in front]]
    return df

def write_scaled_csv_from_raw(raw_dir, norm_factors, additional_columns=None):
    if additional_columns is None:
        additional_columns = []
    df = pd.read_csv(raw_dir)
    for metric in aggregated.keys():
        if metric not in df.columns:
            orig_metric = metric.removesuffix('_' + metric.split('_')[-1])
            df[metric] = df[orig_metric]
        mean, std = norm_factors[metric]
        df[f'{metric}_s'] = (df[metric] - mean) / std 
    df = order_columns(df, additional_columns=additional_columns)
    df.to_csv(f'{raw_dir}.scaled')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    args = parser.parse_args()

    model = args.model
    datasets_to_compute = [
        'filler',
        # 'ClassicGP',
        # 'Agreement',
        # 'RelativeClause',
        # 'AttachmentAmbiguity'
    ]
    additional_columns = ['recovered']

    # Initialize garden path columns first
    aggregated = {}
    df = pd.read_csv(f'out/{model}/items_ClassicGP.parser.csv')
    for metric in INFO_METRICS_TO_SAVE:
        aggregated[metric] = df[metric].tolist()
        for column in additional_columns:
            aggregated[f'{metric}_{column}'] = df[f'{metric}_{column}'].tolist()

    # Add columns from non-GP (use real parses for forcecd recovered columns)
    for dataset in datasets_to_compute:
        df = pd.read_csv(f'out/{model}/items_{dataset}.parser.csv')
        for metric in INFO_METRICS_TO_SAVE:
            aggregated[metric].extend(df[metric].tolist())
            for column in additional_columns:
                aggregated[f'{metric}_{column}'].extend(df[metric].tolist())

    # Get means and std for each predictor type
    norm_factors = {}
    for metric in aggregated.keys():
        norm_factors[metric] = (
            np.mean(aggregated[metric]), 
            np.std(aggregated[metric])
        )

    # Write scaled files
    write_scaled_csv_from_raw(
        raw_dir=f'out/{model}/items_ClassicGP.parser.csv',
        norm_factors=norm_factors,
        additional_columns=additional_columns
    )
    for dataset in datasets_to_compute:
        write_scaled_csv_from_raw(
            raw_dir=f'out/{model}/items_{dataset}.parser.csv',
            norm_factors=norm_factors,
            additional_columns=additional_columns
        )