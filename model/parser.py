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
        incremental,
        mask_next_prob=0.5 
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
        self.next_head_embed = torch.nn.Parameter(
            torch.randn(self.embedding_size)
        )
        self.next_dep_embed = torch.nn.Parameter(
            torch.randn(self.embedding_size)
        )
        self.drop = torch.nn.Dropout(dropout)
        torch.nn.init.kaiming_uniform_(self.W_head)
        torch.nn.init.kaiming_uniform_(self.W_dep)

        # save hyperparams
        self.embedding_model_name = embedding_model_name
        self.reg = reg
        self.learning_rate = learning_rate
        self.potential_clamp = potential_clamp
        self.dropout = dropout
        self.entropy_reg = entropy_reg
        self.incremental = incremental
        self.mask_next_prob = mask_next_prob
        self.save_hyperparameters()

        # path to save predictions
        self.prediction_savepath = None
        self.prediction_masknext = False

    def forward(
        self, 
        sentences, 
        lengths, 
        clamp=False, 
        cutoffs=None, 
        mask_next=False
    ):
        """Get score for edge (i, j) as: 
                
                w_score.T @ ReLU(W_head @ h_i + W_dep @ h_j)

        Args:
            sentences : batch of sentences where each sentence is a list of
                        strings
            lengths : number of nodes in each tree, *including a null initial
                      root node* 
        """
        embeddings = self.embedding_model.get_representations(
            sentences=sentences,
            max_len=max(lengths),
            cutoffs=cutoffs
        )

        head_weights = einsum(self.W_head, embeddings, 'd k, b n d -> b n k')
        dep_weights = einsum(self.W_dep, embeddings, 'd k, b n d -> b n k')
        if mask_next:
            cutoffs = cutoffs if cutoffs is not None else lengths
            mask = torch.arange(max(lengths), device=self.device)[None, :] == cutoffs[:, None]
            head_weights[mask] = self.next_head_embed
            dep_weights[mask] = self.next_dep_embed
        head_weights = self.drop(head_weights)
        dep_weights = self.drop(dep_weights)

        # Broadcast to get all possible head-dep pairs 
        head_weights = rearrange(head_weights, 'b n k -> b n 1 k')
        dep_weights = rearrange(dep_weights, 'b n k -> b 1 n k')
        edge_weights = head_weights + dep_weights # type: ignore | (b, n, n, k)

        # Score 
        edge_weights = torch.relu(edge_weights)
        edge_scores = einsum(self.w_score, edge_weights, 'k, b n m k -> b n m')

        # Shift columns for stability
        column_max = torch.max(edge_scores, dim=-1, keepdim=True)[0]
        edge_scores = edge_scores - column_max

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
        
        mt = MatrixTree(
            scores=edge_scores, 
            lens=lengths-1, 
            multiroot=self.incremental
        )
        return mt, clamp_diff
    
    def predict(self, sentences, lengths, mask_next=False):
        with torch.no_grad():
            mt, _ = self.forward(sentences, lengths, mask_next=mask_next)
        return self._predict(mt, lengths,)

    def _predict(self, mt, lengths):
        with torch.no_grad():
            scores = mt.scores.detach().clone()
            best_trees = mst(scores, mt.mask, multiroot=self.incremental) # type: ignore

            return best_trees

    def get_kl(self, sentences, lengths, cutoffs):
        mt_full, _ = self.forward(
            sentences=sentences,
            lengths=lengths,
            cutoffs=cutoffs
        )
        mt_masked, _ = self.forward(
            sentences=sentences,
            lengths=lengths,
            cutoffs=cutoffs,
            mask_next=True,
        )
        return mt_full.kl(mt_masked)
            
    def _loss(self, mt, gold_trees, clamp_diff):
        log_partition = mt.log_partition
        scores = mt.score(gold_trees)
        marginals = mt.marginals
        
        log_probs = (scores - log_partition).double().mean()
        entropy = (log_partition - (marginals * mt.scores).sum((-1, -2))).mean()
        param_norm = sum(p.norm() ** 2 for p in self.parameters() if p.requires_grad) * self.reg
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
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def on_before_optimizer_step(self, optimizer):
        grads = [param.grad.detach().flatten() 
                 for param in self.parameters() if param.grad is not None]
        if grads:
            grad_norm = torch.linalg.vector_norm(torch.cat(grads))
            self.log('grad_norm', grad_norm, on_step=True, on_epoch=False, prog_bar=False)

    def to(self, device):
        self.embedding_model.to(device)
        return super().to(device)

    def on_train_start(self):
        self.embedding_model.eval()

    def training_step(self, batch, batch_idx):
        sentences = batch["sentences"]
        gold_trees = batch["gold_trees"]
        lengths = batch["lengths"]        
        gold_trees = gold_trees.to(self.device)
        lengths = lengths.to(self.device)

        if self.current_epoch % 5 == 0 and batch_idx == 0:
            mask_next = False
        else:
            mask_next = torch.rand(1).item() < self.mask_next_prob
        
        mt, clamp_diff = self.forward(sentences, lengths, clamp=True, mask_next=mask_next)
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
            tree_acc, node_acc, _ = self._accuracy(gold_trees, y_pred, lengths)
            self.log('train acc', tree_acc)
            self.log('train uas', node_acc, prog_bar=True)

        return loss
    
    def on_validation_start(self):
        self.embedding_model.eval()
        self.drop.eval()
        self.eval()

    def validation_step(self, batch, batch_idx):
        sentences = batch["sentences"]
        gold_trees = batch["gold_trees"]
        lengths = batch["lengths"]        
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
        self.log('val acc', tree_acc)
        self.log('val uas', node_acc, prog_bar=True)

        return loss
    
    def on_test_start(self):
        self.embedding_model.eval()
        self.drop.eval()
        self.eval()
        self.test_predictions = []
        self.cutoffs = []
        self.node_acc = self.node_total = self.tree_acc = self.tree_total = self.probs = 0

    def set_prediction_save_path(self, dir):
        self.prediction_savepath = dir

    def test_step(self, batch, batch_idx):
        with torch.enable_grad():
            sentences = batch["sentences"]
            gold_trees = batch["gold_trees"]
            lengths = batch["lengths"]
            cutoffs = batch["cutoffs"]        
            gold_trees = gold_trees.to(self.device) if gold_trees is not None else None          
            lengths = lengths.to(self.device)
            if cutoffs is None:
                cutoffs = [None for _ in range(len(sentences))]
            else:
                cutoffs = cutoffs.to(self.device)


            mt, _ = self.forward(
                sentences, 
                lengths, 
                clamp=True, 
                cutoffs=cutoffs,
                mask_next= self.prediction_masknext
            )
            y_pred = self._predict(mt, lengths)
            if gold_trees is not None:
                tree_acc, node_acc, _ = self._accuracy(gold_trees, y_pred, lengths)
            else:
                tree_acc = node_acc = 0

            self.test_predictions.extend(zip(sentences, y_pred.cpu().numpy()))
            self.cutoffs.extend(cutoffs)
            self.tree_acc += tree_acc
            self.tree_total += 1
            self.node_acc += node_acc
            self.node_total += 1

    def on_test_end(self):
        tree_acc = self.tree_acc / self.tree_total
        node_acc = self.node_acc / self.node_total
        stat_string = f'acc={tree_acc:.4f}_uas={node_acc:.4f}'
        if self.prediction_savepath is None:
            save_path = self.logger.log_dir + f'/predictions_{stat_string}.conllu' # type: ignore
        
        tensors_to_conllu(
            [sentence for sentence, _ in self.test_predictions],
            [tree for _, tree in self.test_predictions],
            self.cutoffs,
            self.prediction_savepath    
        )
        self.prediction_savepath = None 
    
    def setup(self, stage=None):
        self.embedding_model.to(self.device)

