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

from model.dataset import TreebankDataset, treebank_collater
from model.parser import Parser
from model.utils import build_loader

torch.set_float32_matmul_precision("medium")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT = Path(__file__).resolve().parent
en_train = ROOT / "data" / "treebanks" / "UD_English-GUM" / "en_gum-ud-train-inc.conllu"
en_test = ROOT / "data" / "treebanks" / "UD_English-GUM" / "en_gum-ud-test.conllu"
en_dev = ROOT / "data" / "treebanks" / "UD_English-GUM" / "en_gum-ud-dev.conllu"
en_llm = 'goldfish-models/eng_latn_1000mb'


parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('-train', '--train_dir',
                    default=en_train)
parser.add_argument('-val', '--val_dir',
                    default=en_dev)
parser.add_argument('-e', '--embedding_model',
                    default=en_llm)
parser.add_argument('-mr', '--multiroot', action='store_true')
parser.add_argument('-l', '--llm_layer', type=int, default=7)
parser.add_argument('-dim', '--embedding_dim', type=int, default=512)
parser.add_argument('-r', '--regularization', type=float, default=1e-2)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
parser.add_argument('-c', '--clamp', type=int, default=25)
parser.add_argument('--mlp_drop', type=float, default=0.2)
parser.add_argument('--emb_drop', type=float, default=0.2)
parser.add_argument('-v', '--version_number', type=int, default=None)
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('-n', '--epochs', type=int, default=200)
parser.add_argument('-p', '--patience', type=int, default=50)
parser.add_argument('-er', '--entropy_reg', type=float, default=0)
parser.add_argument('-s', '--split_prob', type=float, default=0.3)
parser.add_argument('-adj', '--predict_adjunct', action='store_true')
parser.add_argument('--val_every_n', type=int, default=5)
parser.add_argument('--local_steps', type=int, default=0)


if __name__ == '__main__':
    args = parser.parse_args()

    model = Parser(
        embedding_model_name=args.embedding_model,
        incremental=args.multiroot,
        reg=args.regularization,
        potential_clamp=args.clamp, 
        learning_rate=args.learning_rate,
        mlp_dropout=args.mlp_drop,
        emb_dropout=args.emb_drop,
        entropy_reg=args.entropy_reg,
        llm_output_layer=args.llm_layer,
        embedding_dim=args.embedding_dim,
        split_trees_prob=args.split_prob,
        local_steps=args.local_steps,
    )

    train_loader = build_loader(
        args.train_dir, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    val_loader = build_loader(
        args.val_dir,
        batch_size=args.batch_size,
        shuffle=False
    )

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
        precision='bf16-mixed',
        callbacks=[
            ModelCheckpoint(
                monitor='val uas',
                mode='max',
                save_top_k=1,
                filename='best_val_{epoch:02d}',
                save_last=False,
            ),
            ModelCheckpoint(
                monitor='cutoff val uas',
                mode='max',
                save_top_k=1,
                filename='best_cutoff_{epoch:02d}',
                save_last=False,
            ),
            ModelCheckpoint(
                filename='last',
                save_top_k=1,
                mode='max',
                monitor='epoch',
            ),
            # EarlyStopping(monitor='val loss', mode='min', patience=args.patience)
        ],
        inference_mode=False,
        gradient_clip_val=5.0,
        gradient_clip_algorithm='norm',
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
