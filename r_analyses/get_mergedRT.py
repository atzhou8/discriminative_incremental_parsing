import argparse

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv')
    parser.add_argument('output_csv')
    parser.add_argument(
        '--cols',
        nargs='+',
        default=['RT']
    )
    parser.add_argument(
        '--sentence-col',
        default='Sentence',
    )
    parser.add_argument(
        '--position-col',
        default='WordPosition',
    )
    parser.add_argument(
        '--window',
        type=int,
        default=3,
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)

    cols = args.cols
    sentence_col = args.sentence_col
    position_col = args.position_col
    window = args.window

    result = df.copy()
    positions = pd.to_numeric(result[position_col]).tolist()
    sentence_col = result[sentence_col].tolist()

    for col in cols:
        values = pd.to_numeric(result[col]).tolist()
        merged_vals = []
        for idx in range(len(values)):
            curr_sum = 0
            valid = True
            for increment in range(window):
                next_idx = idx + increment                
                if (next_idx >= len(values) or
                    sentence_col[next_idx] != sentence_col[idx]):
                    valid = False
                    break
                value = values[next_idx]
                curr_sum += value
            merged_vals.append(curr_sum if valid else float('nan'))

        result[f'{col}_merged'] = merged_vals


    result.to_csv(args.output_csv, index=False)
