import argparse
import torch
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model.parser import Parser
from model.utils import build_loader

torch.set_float32_matmul_precision('medium')

def make_cutoff_transform(spec):
    if spec == 'none' or spec is None:
        return lambda x: None
    elif spec == 'identity':
        return lambda x: x
    elif spec.startswith('shift:'):
        k = int(spec.split(':')[1])
        return lambda x: x + k
    elif spec.startswith('const:'):
        k = int(spec.split(':')[1])
        return lambda x: k * torch.ones_like(x)
    else:
        raise ValueError('Invalid specification for cutoffs')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-v', '--version', type=int, required=True)
    parser.add_argument('-i', '--input_dir', required=True)
    parser.add_argument('-o', '--output_dir', default=None)
    parser.add_argument('--ckpt', default='val')
    parser.add_argument(
        '-d',
        '--dataset_type',
        choices=['treebank', 'phenomena'],
        default="treebank",
    )
    parser.add_argument(
        '-c',
        '--cutoff_transform',
        default='none',
    )
    parser.add_argument(
        '-m',
        '--mask_next',
        action='store_true',
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    assert args.ckpt in ['val', 'mask', 'last']

    ckpt_dir = Path('lightning_logs') / args.name / f'version_{args.version}' / 'checkpoints'
    if args.ckpt == 'last':
        best_ckpt = ckpt_dir / 'last.ckpt'
    else:
        best_ckpt = next(ckpt_dir.glob(f'best_{args.ckpt}_epoch=*.ckpt'))
    print(f'Predicting from {best_ckpt}')

    # Restore model + its hyperparameters from checkpoint
    model = Parser.load_from_checkpoint(
        best_ckpt,
    )
    test_loader = build_loader(
        Path(args.input_dir), 
        args.batch_size,
        num_workers=0,
        dataset_type=args.dataset_type,
        cutoff_fn=make_cutoff_transform(args.cutoff_transform)
    )

    logger = TensorBoardLogger(
        'lightning_logs',
        name=args.name,
        version=args.version
    )

    if args.output_dir is None:
        output_dir = f'out/{args.name}_predictions.conllu'
    else:
        output_dir = args.output_dir


    model.set_prediction_save_path(output_dir)
    model.prediction_masknext = args.mask_next
    trainer = Trainer(accelerator="auto", logger=logger, inference_mode=False)
    trainer.test(model=model, dataloaders=test_loader)
