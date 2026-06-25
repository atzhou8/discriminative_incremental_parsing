import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.datasets import TreebankDataset
from transformers import RobertaForCausalLM, RobertaTokenizer, AutoConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, EarlyStoppingCallback

torch.set_float32_matmul_precision('medium')

ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('name')
parser.add_argument('-m', '--model_name', default='FacebookAI/roberta-large')
parser.add_argument(
    '-train', '--train_dir',
    default=str(ROOT / 'data' / 'treebanks' / 'UD_English-GUM' / 'en_gum-ud-train.conllu'),
)
parser.add_argument(
    '-val', '--val_dir',
    default=str(ROOT / 'data' / 'treebanks' / 'UD_English-GUM' / 'en_gum-ud-dev.conllu'),
)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-6)
parser.add_argument('-b', '--batch_size', type=int, default=8)
parser.add_argument('-n', '--epochs', type=int, default=100)
parser.add_argument('-p', '--patience', type=int, default=100)
parser.add_argument('--accumulate_grad_batches', type=int, default=1)
parser.add_argument('--eval_steps', type=int, default=100)
parser.add_argument('--save_steps', type=int, default=100)


class LMTreebankDataset(TreebankDataset):

    def __init__(self, data_dir, tokenizer):
        super().__init__(data_dir)
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        words = [w.text for w in self.trees[idx].words]
        sentence = ' '.join(words)
        return self.tokenizer(sentence)


if __name__ == '__main__':
    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)

    config = AutoConfig.from_pretrained(args.model_name)
    config.is_decoder = True
    model = RobertaForCausalLM.from_pretrained(
        args.model_name,
        use_safetensors=True,
        trust_remote_code=False,
        config=config
    )

    dataset = LMTreebankDataset(args.train_dir, tokenizer=tokenizer)
    val_dataset = LMTreebankDataset(args.val_dir, tokenizer=tokenizer)
    collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training
    training_args = TrainingArguments(
        output_dir=f'./lightning_logs/{args.name}',
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        logging_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        save_total_limit=1,
        gradient_accumulation_steps=args.accumulate_grad_batches,
        gradient_checkpointing=True,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    trainer.train()
    model.save_pretrained(f'./lightning_logs/{args.name}/final_model')
    tokenizer.save_pretrained(f'./lightning_logs/{args.name}/final_model')



