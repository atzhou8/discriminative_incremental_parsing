import argparse
from pathlib import Path


def concat_conllu_files(input_paths, output_path):
    parts = []
    for input_path in input_paths:
        text = Path(input_path).read_text(encoding='utf-8').rstrip('\n')
        if text:
            parts.append(text)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n\n'.join(parts) + ('\n' if parts else ''), encoding='utf-8')


parser = argparse.ArgumentParser()
parser.add_argument('inputs', nargs='+', help='Input .conllu files to concatenate')
parser.add_argument('-o', '--output', required=True, help='Output .conllu file')


if __name__ == '__main__':
    args = parser.parse_args()
    concat_conllu_files(args.inputs, args.output)