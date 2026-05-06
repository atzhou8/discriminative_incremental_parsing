import argparse
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from model.synsurp_baseline import (
    SynSurpDataset,
    SynSurpRoBERTa,
    synsurp_collator,
    get_vocab_from_text,
)

torch.set_float32_matmul_precision('medium')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT = Path(__file__).resolve().parent
data_dir = ROOT / 'data' / 'treebanks' / 'SUD_English-EWT+GUM+Reddit+LinES+PUD'
ccg_dir = ROOT / 'data' / 'treebanks' / 'ccgbank'
train_sentences = data_dir / 'en_ewtgumredditlinespud-sentences.txt'
train_tags = data_dir / 'en_ewtgumredditlinespud-ccgtags.txt'
dev_tags = ccg_dir / 'dev.stags'
dev_sentences = ccg_dir / 'dev.words'


parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('-train_sent', '--train_sentences', default=train_sentences)
parser.add_argument('-train_tag', '--train_tags', default=train_tags)
parser.add_argument('-val_sent', '--val_sentences', default=dev_sentences)
parser.add_argument('-val_tag', '--val_tags', default=dev_tags)
parser.add_argument('--model_name', default='FacebookAI/roberta-large')
parser.add_argument('--min_word_count', type=int, default=5)
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('-n', '--epochs', type=int, default=100)
parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
parser.add_argument('--accumulate_grad_batches', type=int, default=1)
parser.add_argument('-v', '--version_number', type=int, default=None)

if __name__ == '__main__':
    args = parser.parse_args()

    train_dataset = SynSurpDataset(
        sentence_dir=args.train_sentences,
        tag_dir=args.train_tags,
    )
    val_dataset = SynSurpDataset(
        sentence_dir=args.val_sentences,
        tag_dir=args.val_tags,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=synsurp_collator,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=synsurp_collator,
        num_workers=0,
        pin_memory=True,
    )

    ccg_tags = get_vocab_from_text(args.train_tags)
    words = get_vocab_from_text(
        args.train_sentences,
        min_count=args.min_word_count,
    )

    model = SynSurpRoBERTa(
        model_name=args.model_name,
        ccg_tagset=ccg_tags,
        wordset=words,
        learning_rate=args.learning_rate,
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
                monitor='val loss',
                mode='min',
                save_top_k=1,
                filename='best_val_{epoch:02d}',
                save_last=True,
            ),
            EarlyStopping(
                monitor='val loss',
                mode='min',
                patience=100,
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
