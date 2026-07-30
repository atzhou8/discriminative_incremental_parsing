from typing import Any

import torch
import pytorch_lightning as pl

from supar import LinearChainCRF
from transformers import RobertaTokenizerFast, RobertaModel
from einops import repeat, rearrange


class LinearChainCRFSuperTagger(pl.LightningModule):

    def __init__(
        self,
        model_name,
        ccg_tagset,
        learning_rate,
        split_prob,
        mask_prob,
    ):
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            model_name, 
            add_prefix_space=True,
            local_files_only=True,
        )
        self.model = RobertaModel.from_pretrained(
            model_name,
            use_safetensors=True,
            trust_remote_code=False,
            local_files_only=True,
        )
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        self.hidden_dim = self.model.config.hidden_size
        self.tagset = ccg_tagset
        self.id2tag = {i: t for i, t in enumerate(ccg_tagset)}
        self.tag2id = {t: i for i, t in enumerate(ccg_tagset)}
        self.num_tags = len(ccg_tagset)
        self.learning_rate = learning_rate
        self.split_prob = split_prob
        self.mask_prob = mask_prob

        # tag prediction head
        self.supertagging_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.LayerNorm(
                normalized_shape=self.hidden_dim,
                eps=self.model.config.layer_norm_eps
            ),
            torch.nn.Linear(self.hidden_dim, self.num_tags)
        )

        # tag transition matrix
        self.transitions = torch.nn.Parameter(
            torch.empty(self.num_tags+1, self.num_tags+1)
        )
        torch.nn.init.xavier_uniform_(self.transitions)
        self.save_hyperparameters()
    
    def forward(self, sentences, lengths, cutoffs=None, mask=False):
        """ Output B x L x |C| of log potentials.
        """
        batch_size = len(sentences)
        if cutoffs is not None:
            sentences, lengths = self._split_prefix(sentences, cutoffs, mask)

        tokens = self.tokenizer(
            sentences,
            is_split_into_words=True, 
            return_tensors='pt',
            padding=True,
            truncation=True,
        )
        tokens = tokens.to(self.device)
        roberta_output = self.model(
            **tokens, 
        ).last_hidden_state
        
        # Pool token embeddings to word level
        word_embeddings, _ = self._tokens_to_words(roberta_output, tokens, batch_size)
        emissions = self.supertagging_head(word_embeddings)
        crf = LinearChainCRF(emissions, self.transitions, lengths)

        # OLD: trying to use torch_struct
        # log_potentials (N-1) x C_n+1 x C_n 
        # log_potential(c_i -> c_j) = score(c_j) + trans(c_i, c_j)
        # emissions = rearrange(emissions, 'b n c -> b n c 1')
        # transitions = rearrange(self.tag_transitions, 'curr prev -> 1 1 curr prev')
        # log_potentials = emissions + transitions
        # log_potentials[:, 0, :, :] = emissions[:, 0, :, :]
        # crf = LinearChainCRF(log_potentials, lengths)
        return {
            'crf': crf,
            'cut_sentences': sentences,
            'is_adjunct': None
        }
    
    def training_step(self, batch, batch_idx):
        sentences = batch['sentences']
        lengths = batch['lengths']
        batch_size = len(lengths)
        
        to_split_prefix = torch.rand(1).item() < self.split_prob
        to_mask = torch.rand(1).item() < self.mask_prob
        if to_split_prefix:
            cutoffs = torch.randint(1, lengths.max().item(), size=(batch_size,), device=self.device)
            cutoffs = torch.minimum((cutoffs % (lengths - 1)) + 1, lengths)
        else:
            cutoffs = None

        crf = self.forward(
            sentences=sentences,
            lengths=lengths,
            cutoffs=cutoffs,
            mask=to_mask
        )['crf']

        new_lengths = cutoffs if cutoffs is not None else lengths
        tags = self._tags_to_vector(batch['tags'], new_lengths)
        log_prob = (-crf.log_prob(tags)).mean()
        pred = crf.argmax
        acc = self._tag_accuracy(pred, tags, new_lengths)

        if to_split_prefix:
            self.log('split train log probs', -log_prob, batch_size=batch_size, prog_bar=True)
            self.log('split train acc', acc, batch_size=batch_size)
        else:
            self.log('train log probs', -log_prob, batch_size=batch_size, prog_bar=True)
            self.log('train acc', acc, batch_size=batch_size)
        return log_prob

    def validation_step(self, batch, batch_idx):
        sentences = batch['sentences']
        lengths = batch['lengths']
        batch_size = len(lengths)
        
        to_split_prefix = torch.rand(1).item() < self.split_prob
        to_mask = torch.rand(1).item() < self.mask_prob
        if to_split_prefix:
            cutoffs = torch.randint(1, lengths.max().item(), size=(batch_size,), device=self.device)
            cutoffs = torch.minimum((cutoffs % (lengths - 1)) + 1, lengths)
        else:
            cutoffs = None

        crf = self.forward(
            sentences=sentences,
            lengths=lengths,
            cutoffs=cutoffs,
            mask=to_mask
        )['crf']

        new_lengths = cutoffs if cutoffs is not None else lengths
        tags = self._tags_to_vector(batch['tags'], new_lengths)
        log_prob = (-crf.log_prob(tags)).mean()
        pred = crf.argmax
        acc = self._tag_accuracy(pred, tags, new_lengths)

        self.log('val log probs', -log_prob, batch_size=batch_size, prog_bar=True)
        self.log('val acc', acc, batch_size=batch_size, prog_bar=True)
        return log_prob

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return opt

    def _split_prefix(self, sentences, cutoffs, mask):
        cut_sentences = []
        for i, cutoff in enumerate(cutoffs):
            sentence = sentences[i]
            if mask:
                cut_sentences.append(sentence[:cutoff-1] + ['<mask>'])
            else:
                cut_sentences.append(sentence[:cutoff])

        return cut_sentences, cutoffs

    def _tag_accuracy(self, pred, labels, lengths):
        assert pred.shape == labels.shape
        mask = torch.arange(pred.shape[1], device=pred.device)[None, :] < lengths[:, None]
        correct = (pred.eq(labels) & mask).sum().float()
        total = mask.sum().clamp_min(1).float()
        return correct / total

    def _tags_to_vector(self, tags, lengths):
        batch_size = len(tags)
        max_len = max(lengths)
        tag_vector = torch.zeros((batch_size, max_len), dtype=torch.long)
        for b in range(batch_size):
            tag_seq_ids = [self.tag2id[tag] for tag in tags[b]]
            tag_seq_ids = tag_seq_ids[:lengths[b]]
            tag_seq_ids += [0] * (max_len - len(tag_seq_ids))
            tag_vector[b] = torch.tensor(tag_seq_ids, dtype=torch.long)

        return tag_vector.to(self.device) 

    def _vector_to_tags(self, vector):
        pass
    
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
        valid = (batch_word_ids > 0)
        valid[:, 0] = True 

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
        
        return word_embeddings[:, 1:, :], max_word_id
    




