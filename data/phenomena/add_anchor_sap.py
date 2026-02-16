import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_file')
parser.add_argument('-o', '--output_file', default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    if output_file is None:
        output_file = input_file.replace('.csv', '_anchor.csv')

    df = pd.read_csv(input_file)
    df['sentence'] = '<anchor> ' + df['sentence']
    df['critical_pos'] = df['critical_pos'] + 1
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")