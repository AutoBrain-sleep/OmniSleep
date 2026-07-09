"""Training loops for OmniSleep pretraining phases."""

from __future__ import annotations

from . import distributed as dist_utils
from .config import GRADIENT_ACCUMULATION_STEPS, MAE_MASK_RATIO, TEMPERATURE
from .losses import contrastive_loss_with_limit, masked_contrastive_loss

import torch
from tqdm import tqdm


def unwrap_model(model):
    """Return the underlying module for both plain modules and DDP wrappers."""
    return model.module if hasattr(model, "module") else model


def _autocast_context(device):
    return torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


def train_phase_1_loo_v2(model, loader, optimizer, device, epoch):
    model.train()
    acc_loss = torch.tensor(0.0, device=device)
    acc_brain = torch.tensor(0.0, device=device)
    acc_body = torch.tensor(0.0, device=device)
    steps = 0
    if dist_utils.is_main_process():
        pbar = tqdm(loader, desc=f"P1 (Hybrid) Ep {epoch}", ncols=120)
    else:
        pbar = loader
    for step, (inputs, masks) in enumerate(pbar):
        inputs = {k: v.to(device, non_blocking=True) for k,v in inputs.items()}
        masks = {k: v.to(device, non_blocking=True) for k,v in masks.items()}
        B, _, L, _ = inputs['eeg'].shape
        with _autocast_context(device):
            brain_pairs, body_pair, _ = unwrap_model(model).forward_phase1_v2(inputs, masks)
            l_brain = 0
            for pair in brain_pairs:
                l_brain += contrastive_loss_with_limit(pair[0], pair[1], TEMPERATURE)
            l_brain /= 3.0
            mask_ecg = masks['ecg'].view(B, 1).expand(B, L).reshape(-1)
            mask_resp = masks['resp'].view(B, 1).expand(B, L).reshape(-1)
            l_body = masked_contrastive_loss(
                body_pair[0], body_pair[1], mask_ecg, mask_resp, TEMPERATURE
            )
            loss = l_brain + l_body
            loss_scaled = loss / GRADIENT_ACCUMULATION_STEPS

        loss_scaled.backward()
        acc_loss += loss.detach()
        acc_brain += l_brain.detach()
        acc_body += l_body.detach()
        steps += 1
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if dist_utils.is_main_process() and step % 20 == 0:
            pbar.set_postfix({
                'L': f"{loss.item():.4f}",
                'Br': f"{l_brain.item():.4f}",
                'Bo': f"{l_body.item():.4f}"
            })
    results = {'Loss': acc_loss / steps, 'BrainLoss': acc_brain / steps, 'BodyLoss': acc_body / steps}
    return dist_utils.reduce_dict(results)

def train_phase_2_mae(model, loader, optimizer, device, epoch, pool_window=5, local_loss_weight=0.5):
    model.train()

    acc_loss = torch.tensor(0.0, device=device)
    acc_mae = torch.tensor(0.0, device=device)
    acc_cross = torch.tensor(0.0, device=device)
    acc_local = torch.tensor(0.0, device=device)
    steps = 0

    if dist_utils.is_main_process():
        pbar = tqdm(loader, desc=f"P2 (MAE) Ep {epoch}", ncols=120)
    else:
        pbar = loader

    for step, (inputs, masks) in enumerate(pbar):
        inputs = {k: v.to(device, non_blocking=True) for k,v in inputs.items()}
        masks = {k: v.to(device, non_blocking=True) for k,v in masks.items()}
        B, _, L, _ = inputs['eeg'].shape

        with _autocast_context(device):
            # Unfreeze CNN: freeze_cnn=False
            outputs, cross_pairs, (brain_pairs, body_pair_local) = unwrap_model(model).forward_phase2_mae(
                inputs, mask_ratio=MAE_MASK_RATIO, freeze_cnn=False, pool_window=pool_window
            )

            # 1. MAE Reconstruction Loss
            loss_mae = 0
            for m in unwrap_model(model).modalities:
                pred = outputs['recon'][m]
                target = outputs['targets'][m]
                mask = outputs['mask'][m] # 1 is masked, 0 is keep

                # Mask check
                mask_bool = mask.bool()

                if mask_bool.sum() > 0:
                    loss_m = (pred - target) ** 2
                    loss_m = loss_m.mean(dim=-1) # (B, L)
                    loss_mae += (loss_m * mask).sum() / (mask.sum() + 1e-8)
                else:
                    loss_mae += 0.0

            loss_mae /= len(unwrap_model(model).modalities)

            # 2. Local Internal Loss (Pairwise)
            l_local_brain = 0
            for pair in brain_pairs:
                l_local_brain += contrastive_loss_with_limit(pair[0], pair[1], TEMPERATURE)
            l_local_brain /= 3.0

            mask_ecg = masks['ecg'].view(B, 1).expand(B, L).reshape(-1)
            mask_resp = masks['resp'].view(B, 1).expand(B, L).reshape(-1)
            l_local_body = masked_contrastive_loss(
                body_pair_local[0], body_pair_local[1], mask_ecg, mask_resp, TEMPERATURE
            )
            loss_local = l_local_brain + l_local_body

            # 3. System Contrastive Loss
            num_windows = L // pool_window
            def get_pooled_mask(mod_name):
                return masks[mod_name].view(B, 1).expand(B, num_windows).reshape(-1)

            m_eeg = get_pooled_mask('eeg')
            m_eog = get_pooled_mask('eog')
            m_emg = get_pooled_mask('emg')
            m_ecg = get_pooled_mask('ecg')
            m_resp = get_pooled_mask('resp')

            mask_pairs = [
                (m_eeg, m_resp), (m_eeg, m_ecg),
                (m_eog, m_resp), (m_eog, m_ecg),
                (m_emg, m_resp), (m_emg, m_ecg)
            ]

            l_cross = 0
            for (p1, p2), (m1, m2) in zip(cross_pairs, mask_pairs):
                l_cross += masked_contrastive_loss(p1, p2, m1, m2, TEMPERATURE)

            loss_cross = l_cross / 6.0

            # Total Loss
            loss = loss_mae + 0.2 * loss_cross + local_loss_weight * loss_local
            loss_scaled = loss / GRADIENT_ACCUMULATION_STEPS

        loss_scaled.backward()

        acc_loss += loss.detach()
        acc_mae += loss_mae.detach()
        acc_cross += loss_cross.detach()
        acc_local += loss_local.detach()
        steps += 1

        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if dist_utils.is_main_process() and step % 20 == 0:
            pbar.set_postfix({
                'L': f"{loss.item():.4f}",
                'MAE': f"{loss_mae.item():.4f}",
                'C': f"{loss_cross.item():.4f}",
                'Loc': f"{loss_local.item():.4f}"
            })
            if dist_utils.logger:
                for handler in dist_utils.logger.handlers:
                    handler.flush()

    results = {
        'Loss': acc_loss/steps,
        'MAELoss': acc_mae/steps,
        'CrossLoss': acc_cross/steps,
        'LocalLoss': acc_local/steps
    }
    return dist_utils.reduce_dict(results)
