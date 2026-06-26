from typing import Any

import torch
import pytorch_lightning as pl

from torch_struct import LinearChainCRF, LinearChain
from transformers import RobertaTokenizerFast, RobertaModel
from einops import repeat, rearrange


class LinearChainCRFSuperTagger(pl.LightningModule):

    def __init__(
        self,
        model_name,
        ccg_tagset,
    ):
        super().__init__()
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            model_name, 
            add_prefix_space=True
        )
        self.model = RobertaModel.from_pretrained(
            model_name,
            use_safetensors=True,
            trust_remote_code=False,
        )
        self.hidden_dim = self.model.config.hidden_size
        self.num_tags = len(ccg_tagset)

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
        self.tag_transitions = torch.nn.Parameter(
            torch.empty(self.num_tags, self.num_tags)
        )
        torch.nn.init.xavier_uniform_(self.tag_transitions)
        
    def forward(self, batch):
        """ Output B x L x |C| of log potentials.
        """
        sentences = batch['sentences']
        tags = batch['tags']
        lengths = torch.tensor([len(sentence) for sentence in sentences]).to(self.device)
        batch_size = len(sentences)

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
            output_hidden_states=True
        ).hidden_states[-1]
        
        # Pool token embeddings to word level
        word_embeddings, max_word_id = self._tokens_to_words(roberta_output, tokens, batch_size)
        emissions = self.supertagging_head(word_embeddings)


        # log_potentials (N-1) x C_n+1 x C_n 
        # log_potential(c_i -> c_j) = score(c_j) + trans(c_i, c_j)
        emissions = rearrange(emissions, 'b n c -> b n c 1')
        transitions = rearrange(self.tag_transitions, 'curr prev -> 1 1 curr prev')
        log_potentials = emissions + transitions
        crf = LinearChainCRF(log_potentials, lengths)
    
    def training_step(self, batch, batch_idx):
        crf = self.forward(batch)
        tags = batch['tags']

        log_prob = crf.log_prob


        
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
        word_embeddings[:, 0, :] = token_embeddings[:, 0, :] 
        
        return word_embeddings, max_word_id
    




