import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from einops import rearrange

class EmbeddingModel:
    """Wrapper around a HuggingFace transformers model for retrieving 
    tokenizations and embeddings.
    """

    def __init__(self, model_name, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_safetensors=True,
            trust_remote_code=False,
        ).to(device)
        self.config = self.model.config

    def to(self, device):
        self.device = device
        self.model.to(device)

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
            padding=True,
            truncation=True,
            max_length = max_len
        )
        tokenization.to(self.device)
        return tokenization
    
    def get_representations(self, sentences, max_len, layer=-3):
        """Gets embeddings for each node in a UD tree meaning across subword
        units if necessary. Retrieve embeddings from the last transformer layer
        by default.
        """
        tokenization = self.get_tokenization(sentences, max_len)
        embeddings = self.model(
            **tokenization, 
            output_hidden_states=True
        ).hidden_states[layer] # (batch_size, num_words, embedding_dim)

        # Strip BOS/EOS and combine subwords by meaning across word id
        assert max_len <= len(tokenization.word_ids(0))
        batch_size = len(sentences)
        num_words = max_len
        embedding_dim = self.model.config.hidden_size
        
        batch_word_ids = [tokenization.word_ids(i) for i in range(batch_size)]
        embeddings_cleaned = torch.zeros(
            batch_size,
            num_words,
            embedding_dim,
            device=self.device
        )

        for batch, word_ids in enumerate(batch_word_ids):
            sentence_embed = torch.zeros(num_words, embedding_dim, 
                                         device=self.device)
            sentence_token_counts = torch.zeros(num_words,
                                                device=self.device)
            for embed_id, word_id in enumerate(word_ids):
                if word_id is not None: # ignore BOS/EOS/pad, index starts at 1
                    sentence_embed[word_id+1] += embeddings[batch, embed_id, :]
                    sentence_token_counts[word_id+1] += 1

            token_count_denom = sentence_token_counts.clamp_min(1).unsqueeze(-1)
            sentence_embed = sentence_embed/token_count_denom
            
            sentence_embed[sentence_token_counts == 0] = 0
            embeddings_cleaned[batch, :, :] = sentence_embed

        return embeddings_cleaned
        