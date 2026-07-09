"""Loss functions used by OmniSleep pretraining."""

from __future__ import annotations

from .config import MAX_NEGATIVES

import torch
import torch.nn.functional as F

def contrastive_loss_with_limit(p_anchor, p_positive, temperature=0.1, max_negatives=MAX_NEGATIVES):
    p_anchor = p_anchor.float(); p_positive = p_positive.float()
    batch_size = p_anchor.shape[0]
    logits_pos = torch.einsum('nc,nc->n', [p_anchor, p_positive]).unsqueeze(-1)
    if batch_size > max_negatives:
        neg_indices = torch.randperm(batch_size, device=p_anchor.device)[:max_negatives]
        neg_samples = p_positive[neg_indices]
    else:
        neg_samples = p_positive
    logits_neg = torch.einsum('nc,ck->nk', [p_anchor, neg_samples.T])
    logits = torch.cat([logits_pos, logits_neg], dim=1)
    logits /= temperature
    labels = torch.zeros(batch_size, dtype=torch.long, device=p_anchor.device)
    return F.cross_entropy(logits, labels)

def masked_contrastive_loss(z1, z2, mask1, mask2, temperature=0.1):
    z1, z2 = z1.float(), z2.float()
    sim_matrix = torch.matmul(z1, z2.T) / temperature
    valid_pair_mask = (mask1 * mask2).view(-1)
    if valid_pair_mask.sum() == 0: return torch.tensor(0.0, device=z1.device, requires_grad=True)
    sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - sim_max.detach()
    exp_sim = torch.exp(sim_matrix)
    pos_sim = torch.diag(exp_sim)
    denominator = exp_sim.sum(dim=1)
    log_prob = -torch.log(pos_sim / (denominator + 1e-8))
    loss = (log_prob * valid_pair_mask).sum() / (valid_pair_mask.sum() + 1e-8)
    return loss
