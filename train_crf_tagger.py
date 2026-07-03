import argparse
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from models.lc_crf_tagger import LinearChainCRFSuperTagger
from models.datasets import SupertagDataset, supertag_collator
from models.utils import get_vocab_from_text

torch.set_float32_matmul_precision('medium')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT = Path(__file__).resolve().parent
ccg_dir = ROOT / 'data' / 'treebanks' / 'ccgbank'
train_sentences = ccg_dir / 'train.words'
train_tags = ccg_dir / 'train.stags'
dev_tags = ccg_dir / 'dev.stags'
dev_sentences = ccg_dir / 'dev.words'


parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('-ts', '--train_sentences', default=train_sentences)
parser.add_argument('-tt', '--train_tags', default=train_tags)
parser.add_argument('-vs', '--val_sentences', default=dev_sentences)
parser.add_argument('-vt', '--val_tags', default=dev_tags)
parser.add_argument('--model_name', default='FacebookAI/roberta-large')
parser.add_argument('--min_word_count', type=int, default=1)
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('-n', '--epochs', type=int, default=100)
parser.add_argument('-s', '--split_prob', type=float, default=0.8)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
parser.add_argument('--accumulate_grad_batches', type=int, default=1)
parser.add_argument('-v', '--version_number', type=int, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    ccg_tags = get_vocab_from_text(args.train_tags)
    train_dataset = SupertagDataset(
        sentence_dir=args.train_sentences,
        tag_dir=args.train_tags,
        tagset=ccg_tags,
    )
    val_dataset = SupertagDataset(
        sentence_dir=args.val_sentences,
        tag_dir=args.val_tags,
        tagset=ccg_tags
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=supertag_collator,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=supertag_collator,
        num_workers=0,
        pin_memory=True,
    )

    model = LinearChainCRFSuperTagger(
        model_name=args.model_name,
        ccg_tagset=ccg_tags,
        learning_rate=args.learning_rate,
        split_prob=args.split_prob,
    )
    logger = TensorBoardLogger(
        save_dir='lightning_logs',
        name=args.name,
        version=args.version_number,
    )
    trainer = Trainer(
        accelerator='gpu' if device.type == 'cuda' else 'cpu',
        max_epochs=args.epochs,
        devices=1,
        logger=logger,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision='bf16-mixed',
        callbacks=[
            ModelCheckpoint(
                monitor='val log probs',
                mode='max',
                save_top_k=1,
                filename='best_val_{epoch:02d}',
                save_last=True,
            ),
            ModelCheckpoint(
                filename='last',
                save_top_k=1,
                mode='max',
                monitor='epoch',
            ),
        ],
        inference_mode=False,
        gradient_clip_val=5.0,
        gradient_clip_algorithm='norm',
    )

    print('-' * 80)
    print(f'Training on {args.train_sentences}')
    resume_ckpt = Path(logger.log_dir) / 'checkpoints' / 'last.ckpt'

    if resume_ckpt.exists():
        print(f'Resuming from {resume_ckpt}')
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)
    else:
        print('Initializing new model')
        trainer.validate(model, val_loader)
        trainer.fit(model, train_loader, val_loader)
