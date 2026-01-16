import argparse
from pathlib import Path

import torch
import pandas as pd

from torch.utils.data import DataLoader

from model.parser import Parser
from model.dataset import PhenomenaDataset, phenomena_collater

torch.set_float32_matmul_precision('medium')

"""Assume kl and a csv like in SAP for now, but should make work for full 
passages and metrics in future."""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-v', '--version', type=int, required=True)
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('-o', '--output_dir', default=None)
    parser.add_argument(
        '-d',
        '--dataset_type',
        choices=['treebank', 'phenomena'],
        default="phenomena",
    )
    parser.add_argument('-k', '--get_k_after_cp', type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.output_dir is None:
        output_dir = f'out/{args.name}_kl.csv'
    else:
        output_dir = args.output_dir

    # Restore model + its hyperparameters from checkpoint
    ckpt_dir = Path('lightning_logs') / args.name / f'version_{args.version}' / 'checkpoints'
    print(ckpt_dir)
    best_ckpt = next(ckpt_dir.glob("best_epoch=*.ckpt"))
    model = Parser.load_from_checkpoint(
        best_ckpt,
    )
    model.eval()
    model.to(device)

    loader = DataLoader(
        PhenomenaDataset(args.input_dir),
        batch_size=24,
        shuffle=False,
        collate_fn= phenomena_collater,
    )

    df = {
        'condition': [],
        'sentence': [],
        'kl': [],
    }
    for k in range(1, args.get_k_after_cp+1):
        df[f'kl+{k}'] = []

    for batch in loader:
        sentences = batch['sentences']
        lengths = batch['lengths']
        cutoffs = batch['cutoffs']
        conditions = batch['conditions']
        lengths = lengths.to(device)
        cutoffs = cutoffs.to(device)
       
        kl = model.get_kl(sentences, lengths, cutoffs)
        kl = kl.detach().cpu().numpy()
        df['kl'].extend(kl)

        for k in range(1, args.get_k_after_cp+1):
            kl = model.get_kl(sentences, lengths, cutoffs+k)
            kl = kl.detach().cpu().numpy()
            df[f'kl+{k}'].extend(kl)

        sentences = [' '.join(sentence) for sentence in sentences]
        df['condition'].extend(conditions)
        df['sentence'].extend(sentences)
    pd.DataFrame(df).to_csv(output_dir)