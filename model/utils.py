import torch

from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document
from torch.utils.data import DataLoader
from pathlib import Path

import numpy as np
from .dataset import *
from supar.structs.tree import MatrixTree

def tensors_to_conllu(words, heads, cutoffs, write_path):
    if type(heads) is torch.Tensor:
        heads = heads.detach().cpu().tolist()

    write_path = Path(write_path)
    write_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = len(words)
    sentences = []
    # words = [words[:cutoff] for words, cutoff in zip(words, cutoffs)]
    for b in range(batch_size):
        sentence = []
        for idx in range(len(words[b])): 
            word = words[b][idx]
            head = heads[b][idx+1]
            sentence.append({
                'id': idx+1,
                'text': word,
                'head': head,
            })
        sentences.append(sentence)

    doc = Document(sentences)
    CoNLL.write_doc2conll(doc, write_path)

    return doc

def build_loader(
        path, 
        batch_size, 
        shuffle=False, 
        num_workers=0, 
        dataset_type='treebank',
        cutoff_fn = None
):
    assert dataset_type in ['treebank', 'phenomena'], \
          "Dataset type must be either treebank or phenomena"
    D = TreebankDataset if dataset_type=='treebank' else PhenomenaDataset
    col_fn = (lambda x: treebank_collater(x, cutoff_fn)) if dataset_type=='treebank' \
             else (lambda x: phenomena_collater(x, cutoff_fn))

    return DataLoader(
        D(path),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=col_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

def renyi_entropy(dist, alpha):
    lp = dist.log_partition.detach().cpu().numpy()
    alpha_lp = MatrixTree(
        dist.scores * alpha,
        dist.lens,
        dist.multiroot
    ).log_partition.detach().cpu().numpy()

    return (alpha_lp - alpha * lp) /(1-alpha)

def renyi_divergence(dist_before, dist_after, alpha):
    lp_before = dist_before.log_partition
    lp_after = dist_after.log_partition
    lp_mixed = MatrixTree(
        alpha * dist_before.scores + (1 - alpha) * dist_after.scores,
        dist_before.lens,
        multiroot=dist_before.multiroot
    ).log_partition

    renyi_divergence = (lp_mixed - alpha*lp_before - (1-alpha)*lp_after) / (alpha-1)

    return renyi_divergence.detach().cpu().numpy()

def uniform_dist_like(dist):
    return MatrixTree(
        scores=torch.zeros_like(dist.scores),
        lens=dist.lens,
        multiroot=dist.multiroot
    )

def recovered_dist_like(dist, gold_trees, cutoffs, temp=5):
    scores = dist.scores.clone()
    batch_size = scores.shape[0]

    for batch_idx in range(batch_size):
        tree = gold_trees[batch_idx]
        sentence_len = min(int(dist.lens[batch_idx].item()) + 1, len(tree) + 1)
        batch_cutoff = cutoffs[batch_idx]
        scores[batch_idx, :batch_cutoff+1, :batch_cutoff+1] = -temp
        for dep in range(1, sentence_len):
            head = int(tree[dep - 1])
            if head <= batch_cutoff and dep <= batch_cutoff:
                scores[batch_idx, dep, head] = float(temp)

    return MatrixTree(
        scores=scores,
        lens=dist.lens,
        multiroot=dist.multiroot
    )

def weighted_root_dist_like(dist, scale_factor=5):
    scores = dist.scores.clone()
    scores[:, :, 0] *= scale_factor # b * d * h
    return MatrixTree(
        scores=scores,
        lens=dist.lens,
        multiroot=dist.multiroot
    )

def get_info_metrics(dist_before, dist_after):
    assert (dist_before.lens == dist_after.lens).all()
    renyi_alphas = [2, 3, 5]

    metrics = {}
    metrics['entropy_before'] = dist_before.entropy.detach().cpu().numpy()
    metrics['entropy_after']= dist_after.entropy.detach().cpu().numpy()
    for alpha in renyi_alphas:
        metrics[f'entropy_before_renyi_{alpha}'] = renyi_entropy(dist_before, alpha)
        metrics[f'entropy_after_renyi_{alpha}'] = renyi_entropy(dist_after, alpha)
        metrics[f'renyi_divergence_forward_{alpha}'] = renyi_divergence(dist_before, dist_after, alpha)
        metrics[f'renyi_divergence_backward_{alpha}'] = renyi_divergence(dist_after, dist_before, alpha)
        metrics[f'renyi_divergence_symmetric_{alpha}'] = 0.5 * metrics[f'renyi_divergence_forward_{alpha}'] + 0.5 * metrics[f'renyi_divergence_backward_{alpha}']

    # metrics['entropy_reduction'] = metrics['entropy_before'] - metrics['entropy_after']
    entropy_before = dist_before.entropy.detach().cpu().numpy()
    entropy_after = dist_after.entropy.detach().cpu().numpy()

    metrics['entropy_reduction'] = entropy_before - entropy_after
    metrics['entropy_change'] = np.abs(metrics['entropy_reduction'])
    # metrics['perplexity_before'] = np.exp(metrics['entropy_before'])
    # metrics['perplexity_after'] = np.exp(metrics['entropy_after'])
    metrics['cross_entropy_forward'] = dist_before.cross_entropy(dist_after).detach().cpu().numpy()
    metrics['cross_entropy_backward'] = dist_after.cross_entropy(dist_before).detach().cpu().numpy()
    metrics['kl_forward'] = dist_before.kl(dist_after).detach().cpu().numpy()
    metrics['kl_backward'] = dist_after.kl(dist_before).detach().cpu().numpy()
    metrics['kl_symmetric'] = 0.5 * (metrics['kl_forward'] + metrics['kl_backward'])
    dist_before_rootshift = weighted_root_dist_like(dist_before)
    dist_after_rootshift = weighted_root_dist_like(dist_after)
    metrics['kl_root_shifted'] = dist_before_rootshift.kl(dist_after_rootshift)
    dist_mix = MatrixTree(
        scores=dist_before.scores + dist_after.scores,
        lens=dist_before.lens,
        multiroot=dist_before.multiroot
    )
    js_geo = 0.5 * (dist_before.kl(dist_mix) + dist_after.kl(dist_mix))
    metrics['js_geo'] = js_geo.detach().cpu().numpy()



    return metrics