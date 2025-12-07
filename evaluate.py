import argparse
import torch
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from model.dataset import ParsingDataset, parsing_collater
from model.parser import Parser
from model.utils import build_loader



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-v', '--version', type=int, required=True)
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()



    ckpt_dir = Path('lightning_logs') / args.name / f'version_{args.version}' / 'checkpoints'
    best_ckpt = next(ckpt_dir.glob("best_epoch=*.ckpt"))

    # Restore model + its hyperparameters from checkpoint
    model = Parser.load_from_checkpoint(
        best_ckpt,
        map_location="cpu",
    )
    test_loader = build_loader(Path(args.dataset), args.batch_size)

    trainer = Trainer(accelerator="auto")
    trainer.validate(model=model, dataloaders=test_loader)