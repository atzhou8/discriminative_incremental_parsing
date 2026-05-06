import torch
import torch.nn.functional as f
import pytorch_lightning as pl

from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset
from transformers import AutoConfig, RobertaForCausalLM, RobertaTokenizerFast
from einops import repeat


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
        self.causal_roberta.train(True)

        # supertagging head
        self.num_tags = len(ccg_tagset)
        self.id2tag = {i: t for i, t in enumerate(ccg_tagset)}
        self.tag2id = {t: i for i, t in enumerate(ccg_tagset)}
        self.supertagging_head = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.LayerNorm(
                normalized_shape=(config.hidden_size), 
                eps=config.layer_norm_eps
            ),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(config.hidden_size, self.num_tags)
        )

        # next word (NOT token) head
        self.num_words = len(wordset)
        self.id2word = {i: t for i, t in enumerate(wordset)}
        self.word2id = {t: i for i, t in enumerate(wordset)}        
        self.lm_head = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.LayerNorm(
                normalized_shape=(config.hidden_size), 
                eps=config.layer_norm_eps
            ),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(config.hidden_size, self.num_words)
        )
        self.learning_rate = learning_rate
        
        for param in self.causal_roberta.parameters():
            param.requires_grad = True
        self.causal_roberta.train(True)

        self.save_hyperparameters()

    def forward(self, batch):
        sentences = batch['sentences']
        tags = batch['tags']
        batch_size = len(sentences)

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
        
        # Pool token embeddings to word level
        word_embeddings, max_word_id = self._tokens_to_words(roberta_output, tokens, batch_size)
        
        lm_preds = self.lm_head(word_embeddings)
        supertag_preds = self.supertagging_head(word_embeddings)
              
        word_labels, tag_labels = self._get_word_level_labels(
            sentences,
            tags,
            max_word_id
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
            'tag_loss': tag_loss,
            'word_labels': word_labels,
            'tag_labels': tag_labels,
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

    def _tokens_to_words(self, token_embeddings, tokenized, batch_size):
        """Combines tokens into words by meaning, following same logic as
        embedding_model.py
        """
        batch_word_ids = torch.tensor(
            [[(0 if wid is None else wid + 1) for wid in enc.word_ids]
             for enc in tokenized.encodings],
            dtype=torch.long,
            device=self.device
        )
        valid = batch_word_ids > 0 

        
        max_word_id = batch_word_ids.max().item()
        hidden_size = token_embeddings.shape[-1]        
        word_embeddings = torch.zeros(
            batch_size,
            max_word_id + 1, # type: ignore
            hidden_size,
            device=self.device
        )
        
        emb_ids = repeat(batch_word_ids, 'b n -> b n d', d=hidden_size)
        word_embeddings.scatter_add_(
            dim=1,
            index=emb_ids,
            src=token_embeddings * valid.unsqueeze(-1)
        )        
        counts = torch.zeros(
            batch_size,
            max_word_id + 1, # type: ignore
            device=self.device
        )
        counts.scatter_add_(dim=1, index=batch_word_ids, src=valid.to(counts.dtype))
        
        denom = repeat(counts.clamp_min(1), 'b n -> b n d', d=hidden_size)
        mask = repeat(counts != 0, 'b n -> b n d', d=hidden_size)
        word_embeddings = word_embeddings / denom
        word_embeddings = word_embeddings * mask
        
        # Drop root position carried over from embedding_model logic
        return word_embeddings[:, 1:, :], max_word_id
    
    def _get_word_level_labels(self, sentences, tags, max_word_id):
        """Generate word-level labels for supertagging and next-word prediction.
        Position i corresponds to word i (0-indexed).
        """
        batch_size = len(sentences)
        
        tag_labels = -100 * torch.ones((batch_size, max_word_id), dtype=torch.long, device=self.device)
        word_labels = -100 * torch.ones((batch_size, max_word_id), dtype=torch.long, device=self.device)
        
        for b in range(batch_size):
            curr_tags = tags[b]
            curr_words = sentences[b]
            num_words = len(curr_words)
            
            # Assign labels for each word position (0-indexed)
            for word_idx in range(min(num_words, max_word_id)):
                # Supertag for current word
                curr_tag = curr_tags[word_idx]
                if curr_tag not in self.tag2id:
                    curr_tag = '<oov>'
                tag_labels[b, word_idx] = self.tag2id[curr_tag]
                
                # Next word prediction
                if word_idx + 1 >= len(curr_words):
                    next_word = '<eos>'
                else:
                    next_word = curr_words[word_idx + 1]
                    if next_word not in self.word2id:
                        next_word = '<oov>'
                
                word_labels[b, word_idx] = self.word2id[next_word]
        
        return word_labels.to(self.device), tag_labels.to(self.device)

    def train(self, mode=True):
        super().train(mode)
        self.causal_roberta.train(mode)
        return self
    
    def eval(self):
        super().eval()
        self.causal_roberta.eval()
        return self

    def _get_accuracy(self, preds, labels, ignore_index=-100):
        mask = labels != -100
        total = mask.sum()
        if total.item() == 0:
            return torch.tensor(0.0, device=self.device)
        correct = (preds.eq(labels) & mask).sum().float()
        return correct / total.float()
    
    def on_train_start(self):
        super().on_train_start()
        self.causal_roberta.train(True) 

    def training_step(self, batch, batch_idx):
        batch_size = len(batch['sentences'])
        output = self.forward(batch)
        lm_loss = output['lm_loss']
        tag_loss = output['tag_loss']
        loss = 0 * lm_loss +  tag_loss

        # compute accuracies for next-word LM and supertagging
        word_preds = output['lm_preds'].argmax(dim=-1)
        tag_preds = output['supertag_preds'].argmax(dim=-1)
        word_labels = output['word_labels']
        tag_labels = output['tag_labels']

        word_acc = self._get_accuracy(word_preds, word_labels)
        tag_acc = self._get_accuracy(tag_preds, tag_labels)

        self.log('train loss', loss, prog_bar=True, batch_size=batch_size)
        self.log('train lm loss', lm_loss, batch_size=batch_size)
        self.log('train tag loss', tag_loss, batch_size=batch_size)
        self.log('train word acc', word_acc, batch_size=batch_size)
        self.log('train tag acc', tag_acc, batch_size=batch_size)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_size = len(batch['sentences'])
        output = self.forward(batch)
        lm_loss = output['lm_loss']
        tag_loss = output['tag_loss']
        loss = lm_loss + tag_loss

        # compute accuracies for next-word LM and supertagging
        word_preds = output['lm_preds'].argmax(dim=-1)
        tag_preds = output['supertag_preds'].argmax(dim=-1)
        word_labels = output['word_labels']
        tag_labels = output['tag_labels']

        word_acc = self._get_accuracy(word_preds, word_labels)
        tag_acc = self._get_accuracy(tag_preds, tag_labels)

        self.log('val loss', loss, prog_bar=True, batch_size=batch_size)
        self.log('val lm loss', lm_loss, batch_size=batch_size)
        self.log('val tag loss', tag_loss, batch_size=batch_size)
        self.log('val word acc', word_acc, prog_bar=True, batch_size=batch_size)
        self.log('val tag acc', tag_acc, prog_bar=True, batch_size=batch_size)

        return loss

