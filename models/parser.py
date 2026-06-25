import supar
import torch
import torch.nn.functional as f
import pytorch_lightning as pl
import numpy as np

from einops import einsum, rearrange, repeat
from supar.structs.tree import MatrixTree
from supar.structs.fn import mst

from .embedding_model import EmbeddingModel
from .utils import tensors_to_conllu

class Parser(pl.LightningModule):

    def __init__(
        self, 
        embedding_model_name, 
        learning_rate,
        potential_clamp,
        emb_dropout,
        mlp_dropout,
        entropy_reg,
        incremental,
        llm_output_layer,
        split_trees_prob, 
        embedding_dim=None,
        local_steps=0,
        predict_adjunct=False,
    ):
        super().__init__()
        self.embedding_model = EmbeddingModel(
            embedding_model_name, 
            self.device,
            out_layer = llm_output_layer
        )
        self.llm_dim = self.embedding_model.config.hidden_size
        if embedding_dim is None:
            self.embedding_dim = self.llm_dim // 2
        else:
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

        # adjunct prediction params
        if predict_adjunct:
            self.mlp_adj = torch.nn.Sequential(
                torch.nn.Linear(self.llm_dim, self.embedding_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(mlp_dropout),
                torch.nn.Linear(self.embedding_dim, 1)
            )

        # save hyperparams
        self.embedding_model_name = embedding_model_name
        self.llm_output_layer = llm_output_layer
        self.learning_rate = learning_rate
        self.potential_clamp = potential_clamp
        self.mlp_dropout = mlp_dropout
        self.emb_dropout = emb_dropout
        self.entropy_reg = entropy_reg
        self.multiroot = incremental
        self.split_trees_prob = split_trees_prob
        self.local_steps = local_steps
        self.predict_adjunct = predict_adjunct
        self.save_hyperparameters()

        # path to save predictions
        self.prediction_savepath = None
        self.layer_to_unfreeze = llm_output_layer

    def forward(
        self, 
        sentences, 
        lengths, 
        clamp=False, 
        cutoffs=None,
    ):
        """Get score for edge (i, j) as: 
                
                h.T@ W_pair @ d + w_head.T @ h + w_dep.T @ d + bias

        Args:
            sentences : batch of sentences where each sentence is a list of
                        strings
            lengths : number of nodes in each tree, *including a null initial
                      root node* 
        """
        batch_size = len(sentences)
        embeddings, cut_sentences = self.embedding_model.get_representations(
            sentences=sentences,
            max_len=max(lengths),
            cutoffs=cutoffs,
        )
        embeddings = self.embedding_drop(embeddings)
        
        # adjunct prediction
        is_adjunct = None
        if self.predict_adjunct:
            if cutoffs is not None:
                batch_indices = torch.arange(batch_size, device=embeddings.device)
                next_positions = cutoffs
                in_bounds = next_positions < embeddings.shape[1]
                tokens_to_predict = embeddings[
                    batch_indices[in_bounds], 
                    next_positions[in_bounds]
                ]
                is_adjunct = torch.zeros(batch_size, 1, device=embeddings.device)
                is_adjunct[in_bounds] = self.mlp_adj(tokens_to_predict).to(is_adjunct.dtype)
         
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


        mt = MatrixTree(
            scores=edge_scores, 
            lens=lengths-1, # -1 to ignore root 
            multiroot=self.multiroot
        )
        return mt, clamp_diff, cut_sentences, is_adjunct
    
    def predict(self, sentences, lengths):
        with torch.no_grad():
            mt, _, _, _ = self.forward(sentences, lengths)
        return self._predict(mt, lengths,)

    def _predict(self, mt, lengths):
        with torch.no_grad():
            scores = mt.scores.detach().clone()
            best_trees = mst(scores, mt.mask, multiroot=self.multiroot) # type: ignore

            return best_trees
  
    def _accuracy(self, y, y_pred, lengths):
        mask = torch.arange(y_pred.shape[1], device=self.device)[None, :] < lengths[:, None]
        trees_equal = (y_pred == y) | ~mask
        tree_acc = trees_equal.all(dim=1).float().mean()

        node_matches = ((y_pred == y) & mask).sum()
        node_total = mask.sum().clamp_min(1)
        node_acc = (node_matches / node_total).item()

        return tree_acc, node_acc, node_total

    def _local_loss(self, mt, gold_trees, clamp_diff, lengths):
        batch, num_words, _ = mt.scores.shape
        logits = mt.scores
        log_partition = mt.log_partition
        marginals = mt.marginals
        mask = torch.arange(num_words, device=self.device)[None, :] < lengths[:, None]
        
        logits = logits.view(batch * num_words, num_words)
        targets = gold_trees.view(batch * num_words)
        mask = mask.view(batch * num_words)

        local = f.cross_entropy(logits[mask], targets[mask], reduction='mean')
        entropy = (log_partition - (marginals * mt.scores).sum((-1, -2))).mean()
        loss = local + clamp_diff - self.entropy_reg * entropy
        return loss, clamp_diff, local, entropy

    def _loss(self, mt, gold_trees, clamp_diff):
        log_partition = mt.log_partition
        scores = mt.score(gold_trees)
        marginals = mt.marginals
        
        log_probs = (scores - log_partition).double().mean()
        entropy = (log_partition - (marginals * mt.scores).sum((-1, -2))).mean()
        loss = -log_probs - self.entropy_reg * entropy + clamp_diff
        return loss, clamp_diff, log_probs, entropy
    
    def _adjunct_loss(self, logits, adjunct_labels):
        pos_weight = torch.tensor([6.5], device=logits.device, dtype=logits.dtype)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(
            input=logits.squeeze(),
            target=adjunct_labels.float()
        )
        return loss

    def _adjunct_accuracy(self, logits, adjunct_labels):
        probs = torch.sigmoid(logits.squeeze())
        preds = (probs >= 0.5).long()
        labels = adjunct_labels.long()
        return (preds == labels).float().mean()
    
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
        gold_adjuncts = batch.get('gold_adjuncts')
        cutoffs = batch['cutoffs']     
        gold_trees = gold_trees.to(self.device)
        lengths = lengths.to(self.device)
        batch_size = lengths.shape[0]

        slice_trees = torch.rand(1).item() < self.split_trees_prob
        if slice_trees:
            cutoffs = torch.randint(1, lengths.max().item(), size=(batch_size,), device=self.device)
            cutoffs = torch.minimum(cutoffs % (lengths - 4) + 4, lengths - 2)
        else:
            cutoffs = None      

        mt, clamp_diff, _, is_adjunct = self.forward(
            sentences, 
            lengths, 
            clamp=True, 
            cutoffs=cutoffs
        )
        adjunct_loss = 0
        adjunct_acc = None
        if self.predict_adjunct and gold_adjuncts is not None \
        and cutoffs is not None and is_adjunct is not None:
            batch_indices = torch.arange(batch_size, device=self.device)
            adjunct_labels = gold_adjuncts[batch_indices, cutoffs]
            adjunct_loss = self._adjunct_loss(is_adjunct, adjunct_labels)
            adjunct_acc = self._adjunct_accuracy(is_adjunct, adjunct_labels)
        if self.global_step < self.local_steps:
            num_words = cutoffs if cutoffs is not None else lengths
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
        loss = loss + adjunct_loss

        log_prefix = 'cutoff' if cutoffs is not None else ''
        self.log(f'{log_prefix} train loss', loss, prog_bar=True, batch_size=batch_size)
        self.log(f'{log_prefix} train entropy', entropy, batch_size=batch_size)
        self.log(f'{log_prefix} train probs', probs, batch_size=batch_size)
        self.log(f'{log_prefix} clamp loss', clamp_loss, batch_size=batch_size)
        self.log(f'{log_prefix} train entropy percent', -entropy / loss, batch_size=batch_size)
        self.log(f'{log_prefix} train probs percent', -probs / loss, batch_size=batch_size)
        self.log(f'{log_prefix} clamp loss percent', clamp_loss / loss, batch_size=batch_size)
        if self.predict_adjunct:
            self.log(f'{log_prefix} train adjunct loss', adjunct_loss, batch_size=batch_size)
        if adjunct_acc is not None:
            self.log(f'{log_prefix} train adjunct acc', adjunct_acc, prog_bar=True, batch_size=batch_size)
        self.log('epoch', self.current_epoch, on_epoch=True)


        return loss

    # def slice_prefix(self, gold_trees, cutoffs):
    #     batch_size, num_words = gold_trees.shape
    #     sliced_trees = gold_trees.clone()

    #     # Mask out nodes beyond cutoff
    #     length_mask = torch.arange(num_words, device=self.device)[None, :] > cutoffs[:, None]
    #     sliced_trees[length_mask] = 0

    #     # Set floating nodes to <anchor>
    #     floating_nodes = sliced_trees > cutoffs[:, None] 
    #     sliced_trees[floating_nodes] = 1

    #     return sliced_trees
    
    def on_validation_start(self):
        self.embedding_model.eval()
        self.eval()

    def validation_step(self, batch, batch_idx):
        sentences = batch['sentences']
        gold_trees = batch['gold_trees']
        gold_adjuncts = batch.get('gold_adjuncts')
        lengths = batch['lengths']        
        gold_trees = gold_trees.to(self.device)
        gold_adjuncts = gold_adjuncts.to(self.device) if gold_adjuncts is not None else None
        lengths = lengths.to(self.device)
        batch_size = lengths.shape[0]

        # full-context metrics
        mt, clamp_diff, _, _ = self.forward(sentences, lengths, clamp=True)
        loss, _, probs, entropy = self._loss(mt, gold_trees, clamp_diff)
        y_pred = self._predict(mt, lengths)
        tree_acc, node_acc, _ = self._accuracy(gold_trees, y_pred, lengths)

        self.log('val loss', loss, prog_bar=True, batch_size=batch_size)
        self.log('val entropy', entropy, batch_size=batch_size)
        self.log('val probs', probs, batch_size=batch_size)
        self.log('val entropy percent', -entropy / loss, batch_size=batch_size)
        self.log('val probs percent', -probs / loss, batch_size=batch_size)
        self.log('val acc', tree_acc, batch_size=batch_size)
        self.log('val uas', node_acc, prog_bar=True, batch_size=batch_size)

        # cutoff metrics
        cutoffs = torch.randint(1, lengths.max().item(), size=(batch_size,), device=self.device)
        cutoffs = torch.minimum(cutoffs % (lengths - 4) + 4, lengths - 2)

        mt, clamp_diff, _, is_adjunct = self.forward(
            sentences, 
            lengths, 
            cutoffs=cutoffs, 
            clamp=True
        )
        loss, _, probs, entropy = self._loss(mt, gold_trees, clamp_diff)
        y_pred = self._predict(mt, lengths)
        tree_acc, node_acc, _ = self._accuracy(gold_trees, y_pred, lengths)
        cutoff_adjunct_loss = 0
        cutoff_adjunct_acc = None
        if self.predict_adjunct and gold_adjuncts is not None and is_adjunct is not None:
            batch_indices = torch.arange(batch_size, device=self.device)
            adjunct_labels = gold_adjuncts[batch_indices, cutoffs]
            cutoff_adjunct_loss = self._adjunct_loss(is_adjunct, adjunct_labels)
            cutoff_adjunct_acc = self._adjunct_accuracy(is_adjunct, adjunct_labels)

        self.log('cutoff val loss', loss, prog_bar=True)
        self.log('cutoff val entropy', entropy)
        self.log('cutoff val probs', probs)
        self.log('cutoff val entropy percent', -entropy / loss)
        self.log('cutoff val probs percent', -probs / loss)
        self.log('cutoff val acc', tree_acc)
        self.log('cutoff val uas', node_acc, prog_bar=True)
        if self.predict_adjunct:
            self.log('cutoff val adjunct loss', cutoff_adjunct_loss)
        if cutoff_adjunct_acc is not None:
            self.log('cutoff val adjunct acc', cutoff_adjunct_acc, prog_bar=True)


        return loss
    
    def on_test_start(self):
        self.embedding_model.eval()
        self.eval()
        self.test_predictions = []
        self.cutoffs = []
        self.true_signature = []
        self.node_acc = self.node_total = self.tree_acc = self.tree_total = self.probs = 0
        # track sentences (or examples) that were predicted entirely correctly
        self.correct_examples = []

    def set_prediction_save_path(self, dir):
        self.prediction_savepath = dir

    def test_step(self, batch, batch_idx):
        with torch.enable_grad():
            sentences = batch['sentences']
            gold_trees = batch['gold_trees']
            lengths = batch['lengths']
            raw_cutoffs = batch['cutoffs']
            gold_trees = gold_trees.to(self.device) if gold_trees is not None else None          
            lengths = lengths.to(self.device)
            if raw_cutoffs is None:
                cutoffs = [None for _ in range(len(sentences))]
            else:
                cutoffs = raw_cutoffs.to(self.device)

            mt, clamp_diff, cut_sentences, _ = self.forward(
                sentences, 
                lengths, 
                clamp=True, 
                cutoffs=cutoffs,
            )
            y_pred = self._predict(mt, lengths)
            if gold_trees is not None:
                # compute loss/metrics
                loss, _, probs, entropy = self._loss(mt, gold_trees, clamp_diff)
                lengths = cutoffs if cutoffs[0] is not None else lengths
                tree_acc, node_acc, _ = self._accuracy(gold_trees, y_pred, lengths)
                # compute per-example correctness and record the sentence for correct ones
                try:
                    # ensure numpy arrays for comparison
                    if isinstance(y_pred, np.ndarray):
                        y_np = y_pred
                    else:
                        y_np = y_pred.cpu().numpy()
                    g_np = gold_trees.cpu().numpy()
                    lengths_np = lengths.cpu().numpy() # type: ignore
                    for i in range(y_np.shape[0]):
                        n = y_np.shape[1]
                        mask = np.arange(n) < lengths_np[i]
                        equal_mask = (y_np[i] == g_np[i]) | (~mask)
                        if equal_mask.all():
                            # store the (possibly cut) sentence text
                            self.correct_examples.append(1)
                        else:
                            self.correct_examples.append(0)
                except Exception:
                    pass
            else:
                tree_acc = node_acc = probs = entropy = 0

            self.test_predictions.extend(zip(cut_sentences, y_pred.cpu().numpy())) # type: ignore
            self.cutoffs.extend(cutoffs)
            self.tree_acc += tree_acc
            self.tree_total += 1
            self.node_acc += node_acc
            self.node_total += 1
            self.log('test acc', tree_acc)
            self.log('test uas', node_acc)
            self.log('test probs', probs)
            self.log('test entropy', entropy)

    def on_test_end(self):
        tree_acc = self.tree_acc / self.tree_total
        node_acc = self.node_acc / self.node_total
        # print list of correct test examples
        stat_string = f'acc={tree_acc:.4f}_uas={node_acc:.4f}'
        if self.prediction_savepath is None:
            save_path = self.logger.log_dir + f'/predictions_{stat_string}.conllu' # type: ignore
        
        tensors_to_conllu(
            [sentence for sentence, _ in self.test_predictions],
            [tree for _, tree in self.test_predictions],
            self.prediction_savepath    
        )
        self.prediction_savepath = None 
