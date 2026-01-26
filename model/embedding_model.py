import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from einops import repeat

class EmbeddingModel(torch.nn.Module):
    """Wrapper around a HuggingFace transformers model for retrieving 
    tokenizations and embeddings.
    """

    def __init__(self, model_name, device, out_layer=-6):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_safetensors=True,
            trust_remote_code=False,
        ).to(device)

        num_layers = len(self.model.transformer.h) + 1
        abs_hidden_idx = out_layer if out_layer >= 0 else (num_layers + out_layer)
        layer_to_unfreeze = abs_hidden_idx
        for name, param in self.model.named_parameters():
            if 'lnf' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.out_layer = out_layer
        self.config = self.model.config

    def unfreeze_layer(self, layer):
        if layer < 1:
            return
        for name, param in self.model.named_parameters():
            if name.startswith('transformer.h.'):
                try:
                    block_idx = int(name.split('.')[2])  # transformer.h.{idx}
                except (IndexError, ValueError):
                    block_idx = None
                param.requires_grad = (block_idx is not None and block_idx == layer)
            elif 'lnf' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def to(self, device):
        self.device = device
        return super().to(device)

    def get_tokenization(self, sentences, max_len):
        """Retrieves tokenization scheme given a batch of sentences

        Args:
            sentences : list of sentences, where each sentence is itself a
                        list of words split by UD tokenization
        """
        tokenization = self.tokenizer(
            sentences, 
            is_split_into_words=True, 
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_len
        )
        tokenization.to(self.device)
        return tokenization
    
    def get_representations(self, sentences, max_len, cutoffs=None):
        """Gets embeddings for each node in a UD tree meaning across subword
        units if necessary. Retrieve embeddings from the last transformer layer
        by default.

        TODO: run hyperparam search on layer, but seems like any of the 
        mid-layers are about the same
        """
        # Cutoff sentences for incremental parsing
        if cutoffs is not None:
            sentences = [sentence[:cutoff] for sentence, cutoff in zip(sentences, cutoffs)] 

        tokenization = self.get_tokenization(sentences, max_len)
        # with torch.inference_mode():
        embeddings = self.model(
            **tokenization,
            output_hidden_states=True
        ).hidden_states[self.out_layer]

        # Strip BOS/EOS and combine subwords by meaning across word id
        assert max_len >= len(tokenization.word_ids(0))
        batch_size = len(sentences)
        num_words = max_len
        embedding_dim = self.model.config.hidden_size

        batch_word_ids = torch.tensor(
            [[(0 if wid is None else wid + 1) for wid in enc.word_ids]
            for enc in tokenization.encodings],
            dtype=torch.long,
            device=self.device
        )
        valid = batch_word_ids > 0
        
        embeddings_cleaned = torch.zeros(
            batch_size,
            num_words,
            embedding_dim,
            device=self.device
        )
        emb_ids = repeat(batch_word_ids, 'b n -> b n d', d=embedding_dim)
        embeddings_cleaned.scatter_add_(
            dim=1, 
            index=emb_ids, 
            src=embeddings * valid.unsqueeze(-1)
        )

        counts = torch.zeros(
            batch_size, 
            num_words, 
            device=self.device
        )
        counts.scatter_add_(dim=1, index=batch_word_ids, src=valid.to(counts.dtype))

        denom = repeat(counts.clamp_min(1), 'b n -> b n d', d=embedding_dim)
        mask = repeat(counts != 0, 'b n -> b n d', d=embedding_dim)
        embeddings_cleaned = embeddings_cleaned / denom
        embeddings_cleaned = embeddings_cleaned * mask

        return embeddings_cleaned
        