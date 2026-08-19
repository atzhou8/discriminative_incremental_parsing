import supar
import torch
import torch.nn.functional as f
import pytorch_lightning as pl
import numpy as np

from einops import einsum, rearrange, repeat
from supar.structs.tree import MatrixTree
from supar.structs.fn import mst

from .utils import tensors_to_conllu

import torch

from transformers import RobertaTokenizerFast, RobertaModel
from einops import repeat


class EmbeddingModel(torch.nn.Module):
    """Wrapper around a HuggingFace transformers model for retrieving 
    tokenizations and embeddings.
    """

    def __init__(
        self, 
        model_name, 
        device, 
        use_anchor, 
        pad, 
        out_layer=-1
    ):
        super().__init__()
        self.device = device
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
        ).to(device)

        if use_anchor:
            special_tokens = {'additional_special_tokens': ['<anchor>', '<none>']}
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Make sure all parameters are unfrozen
        for param in self.model.parameters():
            param.requires_grad = True

        self.use_anchor = use_anchor
        self.pad = pad
        self.out_layer = out_layer
        self.config = self.model.config

    def to(self, device):
        self.device = device
        return super().to(device)

    
    def get_representations(self, sentences, max_len, mask_last, cutoffs=None):
        """Gets embeddings for each node in a UD tree meaning across subword
        units if necessary. Retrieve embeddings from the last transformer layer
        by default.

        """
        # Cutoff sentences for incremental parsing
        cut_sentences = []
        for i, sentence in enumerate(sentences):
            original_len = len(sentence)
            if cutoffs is not None:
                cutoff = int(cutoffs[i].item())
                sentence = sentence[:cutoff] 
                if mask_last:
                    sentence[-1] = '<mask>'
            else:
                cutoff = original_len

            if self.pad == 'length':
                num_to_mask = original_len - cutoff
            else:
                num_to_mask = self.pad
            sentence = sentence + ['<mask>']*num_to_mask

            if self.use_anchor:
                sentence = ['<anchor>'] + sentence
            
            cut_sentences.append(sentence)

        # worst case if we just cut off last word (overestimate but it's ok)
        if cutoffs is not None and self.pad != 'length':
            max_len += self.pad
        if self.use_anchor:
            max_len += 1
        tokenization = self.tokenizer(
            cut_sentences, 
            is_split_into_words=True, 
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=max_len+4
        )
        tokenization.to(self.device)
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
        

class Parser(pl.LightningModule):

    def __init__(
        self, 
        embedding_model_name, 
        learning_rate,
        potential_clamp,
        emb_dropout,
        mlp_dropout,
        entropy_reg,
        llm_output_layer,
        split_trees_prob,
        mask_prob, 
        global_norm=1e-4,
        use_anchor=False,
        pad='length', # pad to sentence length if 'length', else pad (int)
        multiroot=False, 
        embedding_dim=512,
        local_steps=0,
    ):
        super().__init__()
        self.embedding_model = EmbeddingModel(
            embedding_model_name, 
            self.device,
            use_anchor=use_anchor,
            pad=pad,
            out_layer = llm_output_layer
        )
        self.llm_dim = self.embedding_model.config.hidden_size
        self.embedding_dim = embedding_dim
        self.embedding_drop = torch.nn.Dropout(emb_dropout)
        
        # parser params
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(self.llm_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(mlp_dropout),
        )
        self.mlp_dep = torch.nn.Sequential(
            torch.nn.Linear(self.llm_dim, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(mlp_dropout),
        )
        self.W_pair = torch.nn.Parameter(
            torch.empty(self.embedding_dim, self.embedding_dim)
        )
        self.w_head = torch.nn.Parameter(torch.zeros(self.embedding_dim))
        self.w_dep = torch.nn.Parameter(torch.zeros(self.embedding_dim))
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.xavier_uniform_(self.W_pair)

        # save hyperparams
        self.embedding_model_name = embedding_model_name
        self.llm_output_layer = llm_output_layer
        self.learning_rate = learning_rate
        self.potential_clamp = potential_clamp
        self.mlp_dropout = mlp_dropout
        self.emb_dropout = emb_dropout
        self.entropy_reg = entropy_reg
        self.mask_prob = mask_prob
        self.split_trees_prob = split_trees_prob
        self.local_steps = local_steps
        self.global_norm_l = global_norm
        
        self.use_anchor = use_anchor
        self.pad = pad
        self.multiroot = use_anchor or multiroot 
        
        self.save_hyperparameters()

        # path to save predictions
        self.prediction_savepath = None
        self.layer_to_unfreeze = llm_output_layer
        self.test_mask_last = True

    def forward(
        self, 
        sentences, 
        lengths,
        mask_last=False, 
        clamp=False, 
        cutoffs=None,
    ):
        """Get score for edge (i, j) as: 
                
                h.T@ W_pair @ d + w_head.T @ h + w_dep.T @ d + bias

            Embedding model handles sentence modification re: adding an
            anchor node and slicing sentences at cutoffs

        Args:
            sentences : batch of sentences where each sentence is a list of
                        strings
            lengths : number of nodes in  full tree, *including a null initial
                      root node* 
            cutoffs : for incremental processing, number of nodes to include
        """
        batch_size = len(sentences)
        embeddings, cut_sentences = self.embedding_model.get_representations(
            sentences=sentences,
            max_len=max(lengths).item(),
            cutoffs=cutoffs,
            mask_last=mask_last
        )
        embeddings = self.embedding_drop(embeddings)
         
        # Parser
        head_repr = self.mlp_head(embeddings)  # (b, n, d)
        dep_repr  = self.mlp_dep(embeddings)   # (b, n, d)

        paired = einsum(
            head_repr, 
            self.W_pair, 
            dep_repr, 
            'b h d, d e, b m e -> b h m'
        )
        head_scores = einsum(self.w_head, head_repr, 'd, b n d -> b n')
        head_scores = rearrange(head_scores, 'b n -> b n 1')
        dep_scores = einsum(self.w_dep, dep_repr, 'd, b n d -> b n')
        dep_scores = rearrange(dep_scores, 'b n -> b 1 n')
        edge_scores = paired + head_scores + dep_scores + self.bias

        # Clamp during training
        if clamp:
            edge_scores_clipped = edge_scores.clamp(
                min=-self.potential_clamp,
                max=self.potential_clamp,
            )
            clamp_diff = torch.abs(edge_scores_clipped - edge_scores)
            clamp_diff = clamp_diff[torch.isfinite(clamp_diff)].sum()
            edge_scores = edge_scores_clipped
        else:
            clamp_diff = 0


        # Adjust number of nodes for matrix tree (length if no cutoff)
        if cutoffs is None or self.pad == 'length':
            num_nodes = lengths
        else:
            num_nodes = cutoffs + self.pad + 1 # +1 for virtual root 
        if self.use_anchor:
            num_nodes += 1


        mt = MatrixTree(
            scores=edge_scores, 
            lens=num_nodes-1, # -1 to ignore virtual root 
            multiroot=self.multiroot
        )

        return {
            'crf': mt,
            'clamp_diff': clamp_diff,
            'cut_sentences': cut_sentences,
        }
    
    def predict(self, sentences, lengths, cutoffs=None):
        with torch.no_grad():
            mt = self.forward(sentences, lengths, cutoffs=cutoffs)['crf']
        return self._predict(mt)

    def _predict(self, mt):
        with torch.no_grad():
            scores = mt.scores.detach().clone()
            best_trees = mst(scores, mt.mask, multiroot=self.multiroot) # type: ignore

            return best_trees
  
    def _accuracy(self, y, y_pred, cutoffs):
        mask = torch.arange(y_pred.shape[1], device=self.device)[None, :] < cutoffs[:, None]
        trees_equal = (y_pred == y) | ~mask
        tree_acc = trees_equal.all(dim=1).float().mean()

        node_matches = ((y_pred == y) & mask).sum()
        node_total = mask.sum().clamp_min(1)
        node_acc = (node_matches / node_total).item()

        return tree_acc, node_acc, node_total

    def _local_loss(self, mt, gold_trees, clamp_diff, cutoffs):
        batch, num_words, _ = mt.scores.shape
        logits = mt.scores
        log_partition = mt.log_partition
        marginals = mt.marginals
        mask = torch.arange(num_words, device=self.device)[None, :] <= cutoffs[:, None]
        
        logits = logits.view(batch * num_words, num_words)
        targets = gold_trees.view(batch * num_words)
        mask = mask.view(batch * num_words)

        local = f.cross_entropy(logits[mask], targets[mask], reduction='mean')
        entropy = (log_partition - (marginals * mt.scores).sum((-1, -2))).mean()
        global_norm = torch.logsumexp(logits[mask], 1)
        global_norm = self.global_norm_l * (global_norm**2).mean()
        loss = local + clamp_diff - self.entropy_reg * entropy + global_norm
        return loss, clamp_diff, local, entropy

    def _loss(self, mt, gold_trees, clamp_diff):
        log_partition = mt.log_partition
        scores = mt.score(gold_trees)
        marginals = mt.marginals
        
        log_probs = (scores - log_partition).double().mean()
        entropy = (log_partition - (marginals * mt.scores).sum((-1, -2))).mean()
        loss = -log_probs - self.entropy_reg * entropy + clamp_diff
        return loss, clamp_diff, log_probs, entropy

    def add_anchor_to_gold_tree(self, gold_trees):
        """Adjusts gold tree labels to fit with new anchor node by 
        incrementing the head of each node."""
        gold_trees = gold_trees.clone()

        # Increment all non-root nodes
        nonroot_mask = (gold_trees != 0)
        gold_trees[nonroot_mask] = gold_trees[nonroot_mask] + 1

        # Insert new column for anchor node
        first_col = torch.zeros(gold_trees.shape[0], 1, device=gold_trees.device)
        gold_trees = torch.hstack((first_col, gold_trees))

        return gold_trees.long()

    def slice_and_pad_trees(self, gold_trees, cutoffs):
        """Revises gold tree labels from treebank to be compatible with 
        incremental prefixes. 

            If using anchor, set all orphaned nodes parent to anchor (<1>)
            Otherwise, set all orphaned nodes parent to root (<0>)

            For both, include true head labels for all self.pad nodes
        """
        sliced_trees = gold_trees.clone()
        if self.pad != 'length':
            cutoffs = cutoffs + self.pad
            extra_pad = torch.zeros(gold_trees.shape[0], self.pad, device=gold_trees.device) # type: ignore
            sliced_trees = torch.hstack((sliced_trees, extra_pad))


        # Mask out nodes beyond cutoff + self.pad
        # Doesn't care if cutoff + pad ends up being longer than original sentence
        # > instead of >= accounts for the extra root node that is not part of sent length
        num_nodes = sliced_trees.shape[1]
        length_mask = torch.arange(num_nodes, device=self.device)[None, :] > cutoffs[:, None]
        sliced_trees[length_mask] = 0

        # Fix surviving orphan nodes
        if self.pad != 'length':
            orphan_head = 1 if self.use_anchor else 0
            floating_nodes = sliced_trees > cutoffs[:, None] 
            sliced_trees[floating_nodes] = orphan_head

        return sliced_trees.long()
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return opt
    
    def on_before_optimizer_step(self, optimizer):
        grads = [param.grad.detach().flatten() 
                 for param in self.parameters() if param.grad is not None]
        if grads:
            grad_norm = torch.linalg.vector_norm(torch.cat(grads))
            self.log('grad_norm', grad_norm, on_step=True, on_epoch=False, prog_bar=False)

    def to(self, device):
        self.embedding_model.to(device)
        return super().to(device)

    def training_step(self, batch, batch_idx):
        sentences = batch['sentences']
        gold_trees = batch['gold_trees']
        lengths = batch['lengths']   
        cutoffs = batch['cutoffs']     
        gold_trees = gold_trees.to(self.device)
        lengths = lengths.to(self.device)
        batch_size = lengths.shape[0]

        # Non-deterministically slice trees
        slice_trees = torch.rand(1).item() < self.split_trees_prob
        if slice_trees:
            cutoffs = torch.randint(
                1, 
                lengths.max().item(), 
                size=(batch_size,), 
                device=self.device
            )
            cutoffs = (cutoffs % (lengths-1)) + 1
            mask_last_word = torch.rand(1).item() < self.mask_prob
        else:
            cutoffs = None
            mask_last_word = False

        # Parser computation
        out = self.forward(
            sentences, 
            lengths, 
            clamp=True,
            mask_last=mask_last_word,
            cutoffs=cutoffs
        )
        mt, clamp_diff = out['crf'], out['clamp_diff']

        # Special case for anchors
        if self.use_anchor:
            gold_trees = self.add_anchor_to_gold_tree(gold_trees)
            cutoffs = cutoffs + 1 if cutoffs is not None else cutoffs
        # Make prefix trees
        if cutoffs is not None:
            gold_trees = self.slice_and_pad_trees(gold_trees, cutoffs)


        # Local loss in begininng of training
        if self.global_step < self.local_steps:
            num_words = cutoffs + self.pad + 1 if cutoffs is not None and self.pad != 'length' else lengths
            loss, clamp_loss, probs, entropy = self._local_loss(
                mt, 
                gold_trees, 
                clamp_diff, 
                num_words
            )
        else:
            loss, clamp_loss, probs, entropy = self._loss(
                mt, 
                gold_trees, 
                clamp_diff
            )
        loss = loss

        log_prefix = 'cutoff' if cutoffs is not None else ''
        self.log(f'{log_prefix} train loss', loss, prog_bar=True, batch_size=batch_size)
        self.log(f'{log_prefix} train entropy', entropy, batch_size=batch_size)
        self.log(f'{log_prefix} train probs', probs, batch_size=batch_size)
        self.log(f'{log_prefix} clamp loss', clamp_loss, batch_size=batch_size)
        self.log(f'{log_prefix} train entropy percent', -entropy / loss, batch_size=batch_size)
        self.log(f'{log_prefix} train probs percent', -probs / loss, batch_size=batch_size)
        self.log(f'{log_prefix} clamp loss percent', clamp_loss / loss, batch_size=batch_size)
        self.log('epoch', self.current_epoch, on_epoch=True)

        return loss
    
    def on_validation_start(self):
        self.embedding_model.eval()
        self.eval()

    def validation_step(self, batch, batch_idx):
        sentences = batch['sentences']
        gold_trees = batch['gold_trees']
        lengths = batch['lengths']   
        cutoffs = batch['cutoffs']     
        gold_trees = gold_trees.to(self.device)
        lengths = lengths.to(self.device)
        batch_size = lengths.shape[0]

        # Non-deterministically slice trees
        slice_trees = torch.rand(1).item() < self.split_trees_prob
        if slice_trees:
            cutoffs = torch.randint(
                1, 
                lengths.max().item(), 
                size=(batch_size,), 
                device=self.device
            )
            cutoffs = (cutoffs % (lengths-1)) + 1
            mask_last_word = torch.rand(1).item() < self.mask_prob
        else:
            cutoffs = None
            mask_last_word = False

        # Parser computation
        out = self.forward(
            sentences, 
            lengths, 
            clamp=True,
            mask_last=mask_last_word,
            cutoffs=cutoffs
        )
        mt, clamp_diff = out['crf'], out['clamp_diff']

        # Special case for anchors
        if self.use_anchor:
            gold_trees = self.add_anchor_to_gold_tree(gold_trees)
            cutoffs = cutoffs + 1 if cutoffs is not None else cutoffs
        # Make prefix trees
        if cutoffs is not None:
            gold_trees = self.slice_and_pad_trees(gold_trees, cutoffs)

        loss, _, probs, entropy = self._loss(mt, gold_trees, clamp_diff)
        y_pred = self._predict(mt)
        num_nodes = lengths if cutoffs is None else cutoffs
        tree_acc, node_acc, _ = self._accuracy(gold_trees, y_pred, num_nodes)

        self.log('val loss', loss, prog_bar=True)
        self.log('val entropy', entropy)
        self.log('val probs', probs)
        self.log('val entropy percent', -entropy / loss)
        self.log('val probs percent', -probs / loss)
        self.log('val acc', tree_acc)
        self.log('val uas', node_acc, prog_bar=True)

        return loss
    
    def on_test_start(self):
        super().on_test_start()
        self.embedding_model.eval()
        self.eval()
        self.test_predictions = []

    def set_prediction_save_path(self, dir):
        self.prediction_savepath = dir

    def set_test_mask_last(self, mask_last):
        self.test_mask_last = mask_last

    def test_step(self, batch, batch_idx):
        with torch.enable_grad():
            sentences = batch['sentences']
            gold_trees = batch['gold_trees']
            lengths = batch['lengths']
            raw_cutoffs = batch['cutoffs']
            lengths = lengths.to(self.device)
            if raw_cutoffs is None:
                cutoffs = None
            else:
                cutoffs = raw_cutoffs.to(self.device)

            out = self.forward(
                sentences, 
                lengths, 
                mask_last=self.test_mask_last,
                cutoffs=cutoffs,
            )
            mt, _, cut_sentences = out['crf'], out['clamp_diff'], out['cut_sentences']
            y_pred = self._predict(mt)
            self.test_predictions.extend(zip(cut_sentences, y_pred.cpu().numpy())) # type: ignore

    def on_test_end(self):
        # print list of correct test examples
        if self.prediction_savepath is None:
            self.prediction_savepath = self.logger.log_dir + '/predictions.conllu' # type: ignore
        
        tensors_to_conllu(
            [sentence for sentence, _ in self.test_predictions],
            [tree for _, tree in self.test_predictions],
            self.prediction_savepath    
        )
        self.prediction_savepath = None