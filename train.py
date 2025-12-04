import os
import platform
import multiprocessing
import argparse
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from model.dataset import ParsingDataset, parsing_collater
from model.parser import Parser

torch.set_float32_matmul_precision("medium")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT = Path(__file__).resolve().parent
en_train = ROOT / "data" / "treebanks" / "UD_English-GUM" / "en_gum-ud-train.conllu"
en_test = ROOT / "data" / "treebanks" / "UD_English-GUM" / "en_gum-ud-test.conllu"
en_dev = ROOT / "data" / "treebanks" / "UD_English-GUM" / "en_gum-ud-dev.conllu"
en_llm = 'goldfish-models/eng_latn_1000mb'


parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('-train', '--train_dir',
                    default=en_train)
parser.add_argument('-val', '--val_dir',
                    default=en_dev)
parser.add_argument('-test', '--test_dir',
                    default=en_test)
parser.add_argument('-e', '--embedding_model',
                    default=en_llm)
parser.add_argument('-r', '--regularization', type=float, default=1e-1)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
parser.add_argument('-c', '--clamp', type=int, default=20)
parser.add_argument('-d', '--dropout', type=float, default=0.6)
parser.add_argument('-v', '--version_number', type=int, default=None)
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('-n', '--epochs', type=int, default=250)
parser.add_argument('-p', '--patience', type=int, default=50)
parser.add_argument('--val_every_n', type=int, default=5)

def build_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=parsing_collater,
        num_workers=0 if platform.system() == "Windows" else 6,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )


if __name__ == '__main__':
    args = parser.parse_args()

    model = Parser(
        embedding_model_name=args.embedding_model,
        reg=args.regularization,
        potential_clamp=args.clamp, 
        learning_rate=args.learning_rate,
        dropout=args.dropout
    )
    train_set = ParsingDataset(args.train_dir)
    test_set = ParsingDataset(args.test_dir)
    val_set = ParsingDataset(args.val_dir)

    train_loader = build_loader(train_set, True)
    val_loader = build_loader(val_set, False)
    test_loader = build_loader(test_set, False)

    logger = TensorBoardLogger(
        'lightning_logs',
        name=args.name,
        version=args.version_number
    )
    

    trainer = Trainer(
        accelerator='gpu' if device.type == 'cuda' else 'cpu',
        max_epochs=args.epochs,
        check_val_every_n_epoch=args.val_every_n,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                monitor='val loss',
                mode='min',
                save_top_k=1,
                filename='best_{epoch:02d}',
                save_last=True,
            ),
            EarlyStopping(monitor='val loss', mode='min', patience=args.patience)
        ],
    )

    print('-' * 80)
    print(f'Training on {args.train_dir}')
    resume_ckpt = Path(logger.log_dir) / 'checkpoints' / 'last.ckpt'
    if resume_ckpt.exists():
        print(f'Resuming model {args.name}, version {args.version_number}')
        trainer.fit(
            model,
            train_loader,
            val_loader,
            ckpt_path=resume_ckpt
        )
    else:
        print('Initializing new model')
        trainer.validate(model, val_loader)
        trainer.fit(
            model,
            train_loader,
            val_loader,
        )
    trainer.test(model=model, dataloaders=val_loader)




