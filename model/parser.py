import supar
import torch
import pytorch_lightning as pl

from einops import einsum, rearrange, repeat
from supar.structs.tree import MatrixTree
from supar.structs.fn import mst

from .embedding_model import EmbeddingModel
from .utils import tensors_to_conllu

class Parser(pl.LightningModule):

    def __init__(
        self, 
        embedding_model_name, 
        reg,
        learning_rate,
        potential_clamp,
        dropout,
        entropy_reg, 
    ):
        super(Parser, self).__init__()
        self.embedding_model = EmbeddingModel(embedding_model_name, self.device)
        self.embedding_size = self.embedding_model.config.hidden_size


        self.projection = torch.nn.Linear(self.embedding_size, self.embedding_size // 2)
        torch.nn.init.kaiming_uniform_(self.projection.weight)

        self.embedding_size = self.embedding_size // 2
        self.W_head = torch.nn.Parameter(
            torch.empty(self.embedding_size, self.embedding_size)
        )
        self.W_dep = torch.nn.Parameter(
            torch.empty(self.embedding_size, self.embedding_size)   
        )
        self.w_score = torch.nn.Parameter(
            torch.randn(self.embedding_size)
        )
    
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.kaiming_uniform_(self.W_head)
        torch.nn.init.kaiming_uniform_(self.W_dep)

        self.reg = reg
        self.entropy_reg = entropy_reg
        self.lr = learning_rate
        self.score_clamp = potential_clamp
        self.save_hyperparameters()

    def forward(self, sentences, lengths, clamp=False):
        """Get score for edge (i, j) as: 
                
                w_score.T @ ReLU(W_head @ h_i + W_dep @ h_j)

        Args:
            sentences : batch of sentences where each sentence is a list of
                        strings
        """
        with torch.no_grad():
            embeddings = self.embedding_model.get_representations(
                sentences,
                max(lengths)
            )

        embeddings = self.projection(embeddings)
        embeddings = torch.relu(embeddings)
        head_weights = einsum(self.W_head, embeddings, 'd k, b n d -> b n k')
        dep_weights = einsum(self.W_dep, embeddings, 'd k, b n d -> b n k')


        # Broadcast to get all possible head-dep pairs 
        head_weights = rearrange(head_weights, 'b n k -> b n 1 k')
        dep_weights = rearrange(dep_weights, 'b n k -> b 1 n k')
        edge_weights = head_weights + dep_weights # type: ignore | (b, n, n, k)
        edge_weights = self.dropout(edge_weights) 

        # Score 
        edge_weights = torch.relu(edge_weights)
        edge_scores = einsum(self.w_score, edge_weights, 'k, b n m k -> b n m')

        # # Shift columns for stability
        column_max = torch.max(edge_scores, dim=-1, keepdim=True)[0]
        edge_scores = edge_scores - column_max

        # Clamp during training
        if clamp:
            edge_scores_clipped = edge_scores.clamp(
                min=-self.score_clamp,
                max=self.score_clamp,
            )
            clamp_diff = torch.abs(edge_scores_clipped - edge_scores)
            clamp_diff = clamp_diff[torch.isfinite(clamp_diff)].sum()
            edge_scores = edge_scores_clipped
        else:
            clamp_diff = 0

        mt = MatrixTree(scores=edge_scores, lens=lengths-1)
        return mt, clamp_diff

    def predict(self, sentences, lengths):
        with torch.no_grad():
            mt, _ = self.forward(sentences, lengths)
        return self._predict(mt, lengths)

    def _predict(self, mt, lengths):
        with torch.no_grad():
            best_trees = mst(mt.scores.detach().clone(), mt.mask) # type: ignore

            return best_trees
            
    def _loss(self, mt, gold_trees, clamp_diff):
        log_partition = mt.log_partition
        scores = mt.score(gold_trees)
        marginals = mt.marginals
        
        log_probs = (scores - log_partition).double().mean()
        entropy = (log_partition - (marginals * mt.scores).sum((-1, -2))).mean()
        param_norm = sum(p.norm() ** 2 for p in self.parameters()) * self.reg
        clamp_loss = clamp_diff * self.reg
        loss = -log_probs - self.entropy_reg * entropy + param_norm + clamp_loss
        return loss, clamp_loss, log_probs, entropy
  
    def _accuracy(self, y, y_pred, lengths):
        mask = torch.arange(y_pred.shape[1], device=self.device)[None, :] < lengths[:, None]
        trees_equal = (y_pred == y) | ~mask
        tree_acc = trees_equal.all(dim=1).float().mean()

        node_matches = ((y_pred == y) & mask).sum()
        node_total = mask.sum().clamp_min(1)
        node_acc = (node_matches / node_total).item()

        return tree_acc, node_acc, node_total
    
    def on_before_optimizer_step(self, optimizer):
        grads = [param.grad.detach().flatten() for param in self.parameters() if param.grad is not None]
        if grads:
            grad_norm = torch.linalg.vector_norm(torch.cat(grads))
            self.log('grad_norm', grad_norm, on_step=True, on_epoch=False, prog_bar=False)

    def training_step(self, batch, batch_idx):
        sentences, gold_trees, lengths = batch
        gold_trees = gold_trees.to(self.device)
        lengths = lengths.to(self.device)

        mt, clamp_diff = self.forward(sentences, lengths, clamp=True)
        loss, clamp_loss, probs, entropy = self._loss(mt, gold_trees, clamp_diff)
        self.log('train loss', loss, prog_bar=True)
        self.log('train entropy', entropy)
        self.log('train probs', probs)
        self.log('clamp loss', clamp_loss)
        self.log('train entropy percent', -entropy / loss)
        self.log('train probs percent', -probs / loss)
        self.log('clamp loss percent', clamp_loss / loss)

        # Get train accuracy once per 5 epochs
        if self.current_epoch % 5 == 0 and batch_idx == 0:
            self.eval()
            with torch.no_grad():
                y_pred = self._predict(mt, lengths)
            self.train()
            tree_acc, node_acc, _ = self._accuracy(gold_trees, y_pred, lengths)
            self.log('train tree acc', tree_acc, prog_bar=True)
            self.log('train node acc', node_acc)

        return loss
    
    def validation_step(self, batch, batch_idx):
        sentences, gold_trees, lengths = batch
        gold_trees = gold_trees.to(self.device)
        lengths = lengths.to(self.device)

        mt, clamp_diff = self.forward(sentences, lengths, clamp=True)
        loss, _, probs, entropy = self._loss(mt, gold_trees, clamp_diff)
        y_pred = self._predict(mt, lengths)
        tree_acc, node_acc, _ = self._accuracy(gold_trees, y_pred, lengths)

        self.log('val loss', loss, prog_bar=True)
        self.log('val entropy', entropy)
        self.log('val probs', probs)
        self.log('val entropy percent', -entropy / loss)
        self.log('val probs percent', -probs / loss)
        self.log('val tree acc', tree_acc, prog_bar=True)
        self.log('val node acc', node_acc)

        return loss
    
    def on_test_start(self):
        self.test_predictions = []
        self.node_acc = self.node_total = self.tree_acc = self.tree_total = self.probs = 0

    def test_step(self, batch, batch_idx):
        sentences, gold_trees, lengths = batch
        gold_trees = gold_trees.to(self.device)
        lengths = lengths.to(self.device)

        mt, clamp_diff = self.forward(sentences, lengths, clamp=True)
        loss, _, probs, entropy = self._loss(mt, gold_trees, clamp_diff)
        y_pred = self._predict(mt, lengths)
        tree_acc, node_acc, node_total = self._accuracy(gold_trees, y_pred, lengths)
       
        self.test_predictions.extend(zip(sentences, y_pred.cpu().numpy()))
        self.tree_acc += tree_acc
        self.tree_total += y_pred.shape[0]
        self.node_acc += node_acc
        self.node_total += node_total

    def on_test_end(self):
        tree_acc = self.tree_acc / self.tree_total
        node_acc = self.node_acc / self.node_total
        stat_string = f'epoch={self.current_epoch}_tacc={tree_acc:.4f}_nacc={node_acc:.4f}'
        tensors_to_conllu(
            [s for s, _ in self.test_predictions],
            [y for _, y in self.test_predictions],
            self.logger.log_dir + f'/predictions_{stat_string}.conllu'
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def setup(self, stage=None):
        self.embedding_model.to(self.device)

