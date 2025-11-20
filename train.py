import os.path
import string
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from torch.utils.data import DataLoader

from model.dataset import ParsingDataset, parsing_collater
from model.parser import Parser
from model.utils import tensors_to_conllu

import os

en_train = '.\\data\\treebanks\\UD_English-GUM\\en_gum-ud-train.conllu'
en_test = '.\\data\\treebanks\\UD_English-GUM\\en_gum-ud-test.conllu'
en_dev = '.\\data\\treebanks\\UD_English-GUM\\en_gum-ud-dev.conllu'
en_llm = 'goldfish-models/eng_latn_1000mb'

parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('-train', '--train_dir',
                    default=en_train)
parser.add_argument('-dev', '--dev_dir',
                    default=en_dev)
parser.add_argument('-test', '--test_dir',
                    default=en_test)
parser.add_argument('-e', '--embedding_model',
                    default=en_llm)
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
parser.add_argument('-n', '--epochs', type=int, default=500)
parser.add_argument('-p', '--patience', type=int, default=50)


if __name__ == '__main__':
    args = parser.parse_args()

    model = Parser(args.embedding_model)
    train_set = ParsingDataset(args.train_dir)
    test_set = ParsingDataset(args.test_dir)
    dev_dir = ParsingDataset(args.dev_dir)

    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=parsing_collater
    )
    test_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=parsing_collater
    )
    dev_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=parsing_collater
    )

    logger = TensorBoardLogger(
        'lightning_logs',
        name=args.name,
    )

    trainer = Trainer(
        # accelerator='gpu',
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[
            # ModelCheckpoint(
            #     monitor='Validation loss',
            #     mode='min',
            #     save_top_k=3,
            #     save_last=True,
            # ),
            # EarlyStopping(monitor='Validation loss', mode='min', patience=args.patience)
        ],
    )

    print('-' * 80)
    print(f'Training on {args.train_dir}')
    trainer.fit(
        model,
        train_loader,
        dev_loader,
    )

    test_batch = next(iter(test_loader))
    sentences, trees, lengths = test_batch
    y_pred = model.predict(sentences, lengths)
    tensors_to_conllu(sentences, y_pred, 'test_run.conllu')


