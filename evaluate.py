import argparse
import torch
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from torch.utils.data import DataLoader

from model.dataset import ParsingDataset, parsing_collater
from model.parser import Parser
from model.utils import build_loader

torch.set_float32_matmul_precision('medium')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-v', '--version', type=int, required=True)
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    ckpt_dir = Path('lightning_logs') / args.name / f'version_{args.version}' / 'checkpoints'
    print(ckpt_dir)
    best_ckpt = next(ckpt_dir.glob("best_epoch=*.ckpt"))

    # Restore model + its hyperparameters from checkpoint
    model = Parser.load_from_checkpoint(
        best_ckpt,
    )
    test_loader = build_loader(
        Path(args.dataset), 
        args.batch_size,
        num_workers=0
    )

    logger = TensorBoardLogger(
        'lightning_logs',
        name=args.name,
        version=args.version
    )

    trainer = Trainer(accelerator="auto", logger=logger, inference_mode=False)
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=str(best_ckpt))
