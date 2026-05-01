import torch
import torch.nn.functional as f
import pytorch_lightning as pl

from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset
from transformers import AutoConfig, RobertaForCausalLM, RobertaTokenizerFast


def get_vocab_from_text(file_name, min_count=1, add_oov=False):
    tag_seqs = Path(file_name).read_text(encoding="utf-8").splitlines()
    counter = Counter()
    for seq in tag_seqs:
        if not seq:
            continue
        counter.update(seq.split())

    tags = [tag for tag, cnt in counter.items() if cnt >= min_count]
    tags.extend(['<bos>', '<eos>', '<oov>'])

    return sorted(tags)


class SynSurpDataset(Dataset):

    def __init__(self, sentence_dir, tag_dir):
        self.all_tags = Path(tag_dir).read_text(encoding="utf-8").splitlines()
        self.all_sentences = Path(sentence_dir).read_text(encoding="utf-8").splitlines()

        assert len(self.all_tags) == len(self.all_sentences)

    def __len__(self):
        return len(self.all_tags)
    
    def __getitem__(self, idx):
        tags = self.all_tags[idx].split(' ')
        words = self.all_sentences[idx].split(' ')
        assert(len(tags) == len(words))
        return words, tags

def synsurp_collator(batch):
    sentences, tags = zip(*batch)
    return {
        'sentences': sentences,
        'tags': tags
    }


class SynSurpRoBERTa(pl.LightningModule):

    def __init__(
        self, 
        model_name, 
        ccg_tagset, 
        wordset,
        learning_rate=5e-5
    ):
        super().__init__()
        # roberta model
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            model_name,
            add_prefix_space=True
        )
        config = AutoConfig.from_pretrained(model_name)
        config.is_decoder = True
        self.causal_roberta = RobertaForCausalLM.from_pretrained(
            model_name,
            use_safetensors=True,
            trust_remote_code=False,
            config=config
        ).roberta

        # supertagging head
        self.num_tags = len(ccg_tagset)
        self.id2tag = {i: t for i, t in enumerate(ccg_tagset)}
        self.tag2id = {t: i for i, t in enumerate(ccg_tagset)}
        self.supertagging_head = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.LayerNorm(
                normalized_shape=(config.hidden_size), 
                eps=config.layer_norm_eps
            ),
            torch.nn.Linear(config.hidden_size, self.num_tags)
        )

        # next word (NOT token) head
        self.num_words = len(wordset)
        self.id2word = {i: t for i, t in enumerate(wordset)}
        self.word2id = {t: i for i, t in enumerate(wordset)}        
        self.lm_head = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.LayerNorm(
                normalized_shape=(config.hidden_size), 
                eps=config.layer_norm_eps
            ),
            torch.nn.Linear(config.hidden_size, self.num_words)
        )
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, batch):
        sentences = batch['sentences']
        tags = batch['tags']

        tokens = self.tokenizer(
            sentences,
            is_split_into_words=True, 
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        tokens = tokens.to(self.device)
        roberta_output = self.causal_roberta(
            **tokens, 
            output_hidden_states=True).hidden_states[-1]
        lm_preds = self.lm_head(roberta_output)
        supertag_preds = self.supertagging_head(roberta_output)
              
        word_labels, tag_labels = self._get_labels_from_tokenization(
            sentences,
            tags,
            tokens
        )
        lm_loss = f.cross_entropy(
            lm_preds.reshape(-1, self.num_words),
            word_labels.reshape(-1)
        )
        tag_loss = f.cross_entropy(
            supertag_preds.reshape(-1, self.num_tags),
            tag_labels.reshape(-1)
        )
        
        return {
            'lm_preds': lm_preds,
            'supertag_preds': supertag_preds,
            'lm_loss': lm_loss,
            'tag_loss': tag_loss
        }

    def configure_optimizers(self):
        head_params = []
        head_params += list(self.lm_head.parameters())
        head_params += list(self.supertagging_head.parameters())
        llm_params = [p for p in self.causal_roberta.parameters()]

        opt = torch.optim.Adam(
            [
                {"params": head_params, "lr": self.learning_rate, "weight_decay": 1e-2},
                {"params": llm_params, "lr": self.learning_rate * 0.01, "weight_decay": 1e-3},
            ],
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def _get_labels_from_tokenization(self, sentences, tags, tokenized):
        batch_size = len(sentences)
        input_ids = tokenized['input_ids']

        # label curr tag and next word at last token of each word
        tag_labels = -100 * torch.ones_like(input_ids)
        word_labels = -100 * torch.ones_like(input_ids)
        for b in range(batch_size):
            curr_tags = tags[b]
            curr_words = sentences[b]
            word_ids = tokenized.word_ids(b)
            word_id_to_last_tok_id = {}

            # populate word -> last token id mapping
            for token_id, word_id in enumerate(word_ids):
                if word_id is not None:
                    word_id_to_last_tok_id[word_id] = token_id

            # insert next word and curr tag labels based on mapping
            for word_id, last_token_id in word_id_to_last_tok_id.items():
                curr_tag = curr_tags[word_id]
                if curr_tag not in self.tag2id:
                    curr_tag = '<oov>'
                tag_labels[b, last_token_id] = self.tag2id[curr_tag]

                if word_id+1 >= len(curr_words):
                    next_word = '<eos>'
                else:
                    next_word = curr_words[word_id+1]
                    if next_word not in self.word2id.keys():
                        next_word = '<oov>'
                
                word_labels[b, last_token_id] = self.word2id[next_word]

        return word_labels.to(self.device), tag_labels.to(self.device)
    
    def training_step(self, batch, batch_idx):
        batch_size = len(batch['sentences'])
        output = self.forward(batch)
        lm_loss = output['lm_loss']
        tag_loss = output['tag_loss']
        loss = lm_loss + tag_loss

        self.log(f'train loss', loss, prog_bar=True, batch_size=batch_size)
        self.log(f'train lm loss', lm_loss, batch_size=batch_size)
        self.log(f'train tag loss', tag_loss, batch_size=batch_size)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_size = len(batch['sentences'])
        output = self.forward(batch)
        lm_loss = output['lm_loss']
        tag_loss = output['tag_loss']
        loss = lm_loss + tag_loss

        self.log(f'val loss', loss, prog_bar=True, batch_size=batch_size)
        self.log(f'val lm loss', lm_loss, batch_size=batch_size)
        self.log(f'val tag loss', tag_loss, batch_size=batch_size)

        return loss

