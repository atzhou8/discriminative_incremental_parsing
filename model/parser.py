import supar
import torch
import torch.nn.functional as f
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
        emb_dropout,
        mlp_dropout,
        entropy_reg,
        incremental,
        llm_output_layer,
        mask_next_prob,
        split_trees_prob, 
        embedding_dim=None,
        local_steps=0,
    ):
        super(Parser, self).__init__()
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
        self.W_pair = torch.nn.Parameter(torch.empty(self.embedding_dim, self.embedding_dim))
        self.w_head = torch.nn.Parameter(torch.zeros(self.embedding_dim))
        self.w_dep = torch.nn.Parameter(torch.zeros(self.embedding_dim))
        self.bias = torch.nn.Parameter(torch.zeros(1))
        torch.nn.init.xavier_uniform_(self.W_pair)

        # save hyperparams
        self.embedding_model_name = embedding_model_name
        self.reg = reg
        self.llm_output_layer = llm_output_layer
        self.learning_rate = learning_rate
        self.potential_clamp = potential_clamp
        self.mlp_dropout = mlp_dropout
        self.emb_dropout = emb_dropout
        self.entropy_reg = entropy_reg
        self.incremental = incremental
        self.mask_next_prob = mask_next_prob
        self.split_trees_prob = split_trees_prob
        self.local_steps = local_steps
        self.save_hyperparameters()

        # path to save predictions
        self.prediction_savepath = None
        self.prediction_masknext = False
        self.layer_to_unfreeze = llm_output_layer

    def forward(
        self, 
        sentences, 
        lengths, 
        clamp=False, 
        cutoffs=None, 
        mask_next=False
    ):
        """Get score for edge (i, j) as: 
                
                h.T@ W_pair @ d + w_head.T @ h + w_dep.T @ h + bias

        Args:
            sentences : batch of sentences where each sentence is a list of
                        strings
            lengths : number of nodes in each tree, *including a null initial
                      root node* 
        """
        embeddings, cut_sentences = self.embedding_model.get_representations(
            sentences=sentences,
            max_len=max(lengths),
            cutoffs=cutoffs,
            mask_next=mask_next,
        )
        embeddings = self.embedding_drop(embeddings)

        # Project token representations
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

        # Shift columns for stability
        # column_max = torch.max(edge_scores, dim=-1, keepdim=True)[0]
        # edge_scores = edge_scores - column_max

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

        if cutoffs is not None and cutoffs[0] is not None:
            lengths = cutoffs + 1
        mt = MatrixTree(
            scores=edge_scores, 
            lens=lengths-1, 
            multiroot=self.incremental
        )
        return mt, clamp_diff, cut_sentences
    
    def predict(self, sentences, lengths, mask_next=False):
        with torch.no_grad():
            mt, _, _ = self.forward(sentences, lengths, mask_next=mask_next)
        return self._predict(mt, lengths,)

    def _predict(self, mt, lengths):
        with torch.no_grad():
            scores = mt.scores.detach().clone()
            best_trees = mst(scores, mt.mask, multiroot=self.incremental) # type: ignore

            return best_trees

    def get_kl(self, sentences, lengths, cutoffs):
        mt_full, _, _ = self.forward(
            sentences=sentences,
            lengths=lengths,
            cutoffs=cutoffs
        )
        mt_masked, _, _ = self.forward(
            sentences=sentences,
            lengths=lengths,
            cutoffs=cutoffs,
            mask_next=True,
        )
        return mt_full.kl(mt_masked)

    def get_perplexity(self, sentences, lengths, cutoffs, mask_next=False):
        mt, _, _ = self.forward(
            sentences=sentences,
            lengths=lengths,
            cutoffs=cutoffs,
            mask_next=mask_next
        )

        return torch.exp(mt.entropy)
  
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
        mask = torch.arange(num_words, device=self.device)[None, :] < lengths[:, None]
        
        logits = logits.view(batch * num_words, num_words)
        targets = gold_trees.view(batch * num_words)
        mask = mask.view(batch * num_words)

        local = f.cross_entropy(logits[mask], targets[mask], reduction="mean")
        entropy = (log_partition - (marginals * mt.scores).sum((-1, -2))).mean()
        clamp_loss = clamp_diff * self.reg
        loss = local + clamp_loss - self.entropy_reg * entropy
        return loss, clamp_loss, local, entropy

    def _loss(self, mt, gold_trees, clamp_diff):
        log_partition = mt.log_partition
        scores = mt.score(gold_trees)
        marginals = mt.marginals
        
        log_probs = (scores - log_partition).double().mean()
        entropy = (log_partition - (marginals * mt.scores).sum((-1, -2))).mean()
        clamp_loss = 0
        # clamp_loss = clamp_diff * self.reg
        loss = -log_probs - self.entropy_reg * entropy + clamp_loss
        return loss, clamp_loss, log_probs, entropy
    
    def configure_optimizers(self):
        parser_params = []
        parser_params += list(self.mlp_head.parameters())
        parser_params += list(self.mlp_dep.parameters())
        parser_params += [self.W_pair, self.w_head, self.w_dep, self.bias]
        llm_params = [p for p in self.embedding_model.parameters()]

        opt = torch.optim.Adam(
            [
                {"params": parser_params, "lr": self.learning_rate, "weight_decay": 1e-2},
                {"params": llm_params, "lr": self.learning_rate * 0.01, "weight_decay": 1e-3},
            ],
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
    
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
        sentences = batch["sentences"]
        gold_trees = batch["gold_trees"]
        lengths = batch["lengths"]        
        gold_trees = gold_trees.to(self.device)
        lengths = lengths.to(self.device)
        batch_size = lengths.shape[0]

        if self.global_step > 1000 and self.global_step % 1000 == 0:
            self.embedding_model.unfreeze_layer(self.layer_to_unfreeze)
            self.layer_to_unfreeze = self.layer_to_unfreeze - 1

        slice_trees = torch.rand(1).item() < self.split_trees_prob

        if slice_trees:
            mask_next = torch.rand(1).item() < self.mask_next_prob
            cutoffs = torch.randint(1, lengths.max().item(), size=(batch_size,), device=self.device)
            cutoffs = cutoffs % (lengths - 1) + 3
            cutoffs = torch.min(cutoffs, lengths - 1)
            gold_trees = self.slice_prefix(gold_trees, cutoffs)
        else:
            mask_next = False
            cutoffs = None      

        mt, clamp_diff, _ = self.forward(sentences, lengths, clamp=True, cutoffs=cutoffs, mask_next=mask_next)
        if self.global_step < self.local_steps:
            num_words = cutoffs if cutoffs is not None else lengths
            loss, clamp_loss, probs, entropy = self._local_loss(mt, gold_trees, clamp_diff, num_words)
        else:
            loss, clamp_loss, probs, entropy = self._loss(mt, gold_trees, clamp_diff)

        log_prefix = 'masked' if mask_next else ''
        self.log(f'{log_prefix} train loss', loss, prog_bar=True)
        self.log(f'{log_prefix} train entropy', entropy)
        self.log(f'{log_prefix} train probs', probs)
        self.log(f'{log_prefix} clamp loss', clamp_loss)
        self.log(f'{log_prefix} train entropy percent', -entropy / loss)
        self.log(f'{log_prefix} train probs percent', -probs / loss)
        self.log(f'{log_prefix} clamp loss percent', clamp_loss / loss)
        self.log('epoch', self.current_epoch, on_epoch=True)


        return loss

    def slice_prefix(self, gold_trees, cutoffs):
        batch_size, num_words = gold_trees.shape
        sliced_trees = gold_trees.clone()

        # Mask out nodes beyond cutoff
        length_mask = torch.arange(num_words, device=self.device)[None, :] > cutoffs[:, None]
        sliced_trees[length_mask] = 0

        # Set floating nodes to <anchor>
        floating_nodes = sliced_trees > cutoffs[:, None]
        sliced_trees[floating_nodes] = 1

        return sliced_trees
    
    def on_validation_start(self):
        self.embedding_model.eval()
        self.eval()

    def validation_step(self, batch, batch_idx):
        sentences = batch["sentences"]
        gold_trees = batch["gold_trees"]
        lengths = batch["lengths"]        
        gold_trees = gold_trees.to(self.device)
        lengths = lengths.to(self.device)
        batch_size = lengths.shape[0]

        # unmasked metrics
        mt, clamp_diff, _ = self.forward(sentences, lengths, clamp=True)
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

        # masked metrics
        cutoffs = torch.randint(1, lengths.max().item(), size=(batch_size,), device=self.device)
        cutoffs = cutoffs % (lengths - 1) + 3
        cutoffs = torch.min(cutoffs, lengths - 1)
        gold_trees = self.slice_prefix(gold_trees, cutoffs)

        mt, clamp_diff, _ = self.forward(sentences, lengths, cutoffs=cutoffs, mask_next=True, clamp=True)
        loss, _, probs, entropy = self._loss(mt, gold_trees, clamp_diff)
        y_pred = self._predict(mt, lengths)
        tree_acc, node_acc, _ = self._accuracy(gold_trees, y_pred, lengths)
        self.log('masked val loss', loss, prog_bar=True)
        self.log('masked val entropy', entropy)
        self.log('masked val probs', probs)
        self.log('masked val entropy percent', -entropy / loss)
        self.log('masked val probs percent', -probs / loss)
        self.log('masked val acc', tree_acc)
        self.log('masked val uas', node_acc, prog_bar=True)


        return loss
    
    def on_test_start(self):
        self.embedding_model.eval()
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
            raw_cutoffs = batch["cutoffs"]        
            gold_trees = gold_trees.to(self.device) if gold_trees is not None else None          
            lengths = lengths.to(self.device)
            if raw_cutoffs is None:
                cutoffs = [None for _ in range(len(sentences))]
            else:
                cutoffs = raw_cutoffs.to(self.device)


            mt, clamp_diff, cut_sentences = self.forward(
                sentences, 
                lengths, 
                clamp=True, 
                cutoffs=cutoffs,
                mask_next= self.prediction_masknext
            )
            y_pred = self._predict(mt, lengths)
            if gold_trees is not None:
                if raw_cutoffs is not None:
                    gold_trees = self.slice_prefix(gold_trees, cutoffs)
                    effective_lengths = cutoffs + 1
                else:
                    effective_lengths = lengths
                loss, _, probs, entropy = self._loss(mt, gold_trees, clamp_diff)
                tree_acc, node_acc, _ = self._accuracy(gold_trees, y_pred, effective_lengths)
            else:
                tree_acc = node_acc = probs = entropy = 0

            self.test_predictions.extend(zip(cut_sentences, y_pred.cpu().numpy()))
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
