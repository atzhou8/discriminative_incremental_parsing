import supar
import torch
import pytorch_lightning as pl

from einops import einsum, rearrange, repeat
from supar.structs.tree import MatrixTree
from supar.structs.fn import mst

from .embedding_model import EmbeddingModel

class Parser(pl.LightningModule):

    def __init__(
        self, 
        embedding_model_name, 
        prob_reg=1e-4,
        learning_rate=1e-4,
        potential_clamp=10, 
    ):
        super(Parser, self).__init__()
        self.embedding_model = EmbeddingModel(embedding_model_name, self.device)
        self.embedding_size = self.embedding_model.config.hidden_size

        self.W_head = torch.nn.Parameter(
            torch.empty(self.embedding_size, self.embedding_size)
        )
        self.W_dep = torch.nn.Parameter(
            torch.empty(self.embedding_size, self.embedding_size)   
        )
        self.w_score = torch.nn.Parameter(
            torch.randn(self.embedding_size)
        )
        torch.nn.init.kaiming_uniform_(self.W_head)
        torch.nn.init.kaiming_uniform_(self.W_dep)

        self.prob_reg = prob_reg
        self.lr = learning_rate
        self.score_clamp = potential_clamp

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def setup(self, stage=None):
        self.embedding_model.to(self.device)

    def predict(self, sentences, lengths):
        with torch.no_grad():
            edge_scores = self.forward(sentences, lengths)
            mt = MatrixTree(scores=edge_scores, lens=lengths-1)
            best_trees = mst(mt.scores.detach().clone(), mt.mask) # type: ignore

            return best_trees
        
    def get_loss(self, sentences, gold_trees, lengths):
        edge_scores = self.forward(sentences, lengths)
        
        # clamp for stability
        edge_scores_clipped = edge_scores.clamp(
            min=-self.score_clamp, 
            max=self.score_clamp
        )
        mt = MatrixTree(scores=edge_scores_clipped, lens=lengths-1)
        
        # regularize loss via param norm and clip loss
        gold_trees.to(self.device)
        log_probs = mt.log_prob(gold_trees).double().sum()
        param_norm = sum(p.norm()**2 for p in self.parameters())
        weight_diff = torch.abs(edge_scores_clipped - edge_scores)
        weight_loss = weight_diff[torch.isfinite(weight_diff)].sum()
        loss = -self.prob_reg * log_probs + param_norm + weight_loss
        return loss, weight_loss, log_probs

    def training_step(self, batch, batch_idx):
        sentences, gold_trees, lengths = batch
        gold_trees = gold_trees.to(self.device)
        lengths = lengths.to(self.device)

        loss, weights, probs = self.get_loss(sentences, gold_trees, lengths)
        self.log('train loss', loss, prog_bar=True)
        self.log('train weight diff', weights)
        self.log('train probs', probs)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sentences, gold_trees, lengths = batch
        gold_trees = gold_trees.to(self.device)
        lengths = lengths.to(self.device)

        loss, _, probs = self.get_loss(sentences, gold_trees, lengths)
        self.log('val loss', loss, prog_bar=True)
        self.log('val probs', probs)
        return loss

    def forward(self, sentences, lengths):
        """Get score for edge (i, j) as: 
                (w_score.T @ ReLU(W_head @ h_i + W_dep @ h_j))

        Args:
            sentences : batch of sentences where each sentence is a list of
                        strings
        """
        self.embedding_model.to(self.device)
        with torch.no_grad():
            embeddings = self.embedding_model.get_representations(
                sentences,
                max(lengths)
            )

        head_weights = einsum(self.W_head, embeddings, 'd k, b n d -> b n k')
        dep_weights = einsum(self.W_dep, embeddings, 'd k, b n d -> b n k')

        # Broadcast to get all possible head-dep pairs 
        head_weights = rearrange(head_weights, 'b n k -> b n 1 k')
        dep_weights = rearrange(dep_weights, 'b n k -> b 1 n k')
        edge_weights = head_weights + dep_weights # type: ignore | (b, n, n, k) 

        # Score 
        edge_weights = torch.relu(edge_weights) 
        edge_scores = einsum(self.w_score, edge_weights, 'k, b n m k -> b n m')

        # Mask out self-connections (null nodes handled by supar)
        self_arc_mask = torch.eye(
            edge_scores.size(1),
            device=self.device,
            dtype=torch.bool
        ).unsqueeze(0)
        edge_scores = edge_scores.masked_fill(
            self_arc_mask, 
            float('-inf')
        )

        # Shift columns for stability
        column_max = torch.max(edge_scores, dim=-1, keepdim=True)[0]
        edge_scores = edge_scores - column_max

        return edge_scores
    