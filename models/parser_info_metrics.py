import torch
import numpy as np

from supar.structs.tree import MatrixTree

def renyi_entropy(dist, alpha):
    lp = dist.log_partition.detach().cpu().numpy()
    alpha_lp = MatrixTree(
        dist.scores * alpha,
        dist.lens,
        dist.multiroot
    ).log_partition.detach().cpu().numpy()  # type: ignore

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

def renyi_cross_entropy(dist_before, dist_after, alpha):
    lp_before = dist_before.log_partition
    lp_after = dist_after.log_partition
    lp_mixed = MatrixTree(
        dist_before.scores + (alpha - 1) * dist_after.scores,
        dist_before.lens,
        multiroot=dist_before.multiroot
    ).log_partition

    renyi_divergence = lp_after + (lp_mixed - lp_before) / (1-alpha)

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
        sentence_len = min(int(dist.lens[batch_idx].item()), len(tree))
        batch_cutoff = int(cutoffs[batch_idx].item())
        scores[batch_idx, :batch_cutoff+1, :batch_cutoff+1] = -temp
        for dep in range(1, sentence_len):
            head = int(tree[dep].item()) if isinstance(tree[dep], torch.Tensor) else int(tree[dep])
            if head < 0:
                continue
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
    renyi_alphas = [2, 3, 4, 5, 6]

    metrics = {}
    metrics['entropy_before'] = dist_before.entropy.detach().cpu().numpy()
    metrics['entropy_after']= dist_after.entropy.detach().cpu().numpy()
    for alpha in renyi_alphas:
        metrics[f'entropy_before_renyi_{alpha}'] = renyi_entropy(dist_before, alpha)
        metrics[f'entropy_after_renyi_{alpha}'] = renyi_entropy(dist_after, alpha)
        metrics[f'renyi_divergence_forward_{alpha}'] = renyi_divergence(dist_before, dist_after, alpha)
        metrics[f'renyi_divergence_backward_{alpha}'] = renyi_divergence(dist_after, dist_before, alpha)
        metrics[f'renyi_divergence_symmetric_{alpha}'] = 0.5 * metrics[f'renyi_divergence_forward_{alpha}'] + 0.5 * metrics[f'renyi_divergence_backward_{alpha}']
        metrics[f'renyi_crossent_forward_{alpha}'] = renyi_cross_entropy(dist_before, dist_after, alpha)
        metrics[f'renyi_crossent_backward_{alpha}'] = renyi_cross_entropy(dist_after, dist_before, alpha)
        metrics[f'renyi_crossent_symmetric_{alpha}'] = 0.5 * metrics[f'renyi_crossent_forward_{alpha}'] + 0.5 * metrics[f'renyi_crossent_backward_{alpha}']

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
    dist_mix = MatrixTree(
        scores=dist_before.scores + dist_after.scores,
        lens=dist_before.lens,
        multiroot=dist_before.multiroot
    )
    js_geo = 0.5 * (dist_before.kl(dist_mix) + dist_after.kl(dist_mix))
    metrics['js_geo'] = js_geo.detach().cpu().numpy()

    return metrics