import torch

from transformers import RobertaTokenizerFast, RobertaModel
from einops import repeat

class EmbeddingModel(torch.nn.Module):
    """Wrapper around a HuggingFace transformers model for retrieving 
    tokenizations and embeddings.
    """

    def __init__(self, model_name, device, out_layer=-6):
        super().__init__()
        self.device = device
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            model_name, 
            add_prefix_space=True
        )
        self.model = RobertaModel.from_pretrained(
            model_name,
            use_safetensors=True,
            trust_remote_code=False,
        ).to(device)

        # special_tokens = {'additional_special_tokens': ['<anchor>']}
        # self.tokenizer.add_special_tokens(special_tokens)
        # self.model.resize_token_embeddings(len(self.tokenizer))

        # Make sure all parameters are unfrozen
        for param in self.model.parameters():
            param.requires_grad = True

        self.out_layer = out_layer
        self.config = self.model.config

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

        """
        # Cutoff sentences for incremental parsing
        cut_sentences = []
        for i, sentence in enumerate(sentences):
            if cutoffs is not None:
                cutoff = int(cutoffs[i].item())
                cutoff = max(0, min(cutoff, len(sentence)))
                num_to_mask = len(sentence) - cutoff
                sentence[cutoff:] = ['<mask>'] * num_to_mask
            cut_sentences.append(sentence)

        tokenization = self.get_tokenization(cut_sentences, max_len)
        # with torch.inference_mode():
        embeddings = self.model(
            **tokenization,
            output_hidden_states=True
        ).hidden_states[self.out_layer]

        # Strip BOS/EOS and combine subwords by meaning across word id
        actual_len = len(tokenization.word_ids(0))
        # assert max_len < actual_len
        batch_size = len(cut_sentences)
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
        embeddings_cleaned[:, 0, :] = embeddings[:, 0, :] # splice <s> into root 

        return embeddings_cleaned, cut_sentences
        