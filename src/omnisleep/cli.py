"""Command line entry point for OmniSleep pretraining."""

from __future__ import annotations

from .config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EMB_DIM,
    EPOCHS_PHASE1,
    EPOCHS_PHASE2,
    LR_PHASE1,
    LR_PHASE2,
    LOCAL_LOSS_WEIGHT,
    NUM_WORKERS,
    POOL_WINDOW,
    PREFETCH_FACTOR,
    PRETRAINED_P1_PATH,
    PRETRAINED_P2_PATH,
    SEQ_LEN,
    SKIP_PHASE1,
    SKIP_PHASE2,
    TRAIN_LIST_PATH,
    USE_CHANNELS_LAST,
    USE_SYNC_BN,
    USE_TORCH_COMPILE,
)
from .data import TextListDataset
from .distributed import cleanup_distributed, init_distributed_mode, is_main_process, log_print, setup_logger
from .models import FiveModalSleepModel
from .training import train_phase_1_loo_v2, train_phase_2_mae, unwrap_model

import gc
import os

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _checkpoint_map_location(device: torch.device) -> torch.device:
    return device if device.type == "cuda" else torch.device("cpu")


def _load_checkpoint_if_available(model, checkpoint_path, map_location, label):
    if checkpoint_path and os.path.exists(checkpoint_path):
        log_print(f"\n[Config] Loading {label} Weights: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        unwrap_model(model).load_state_dict(checkpoint, strict=False)
        log_print(f"-> {label} Weights Loaded.")
    elif checkpoint_path:
        log_print(f"\n[Warning] {label} Path not found: {checkpoint_path}")


def main() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    is_distributed, rank, world_size, gpu = init_distributed_mode()
    setup_logger(CHECKPOINT_DIR, rank)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")
        log_print("[Warning] CUDA is not available; training will run on CPU and may be very slow.")

    log_print(f"Batch Size: {BATCH_SIZE} | Workers: {NUM_WORKERS}")

    try:
        train_dataset = TextListDataset(TRAIN_LIST_PATH, seq_len=SEQ_LEN)
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        loader_kwargs = {
            "dataset": train_dataset,
            "batch_size": BATCH_SIZE,
            "sampler": train_sampler,
            "num_workers": NUM_WORKERS,
            "drop_last": True,
            "pin_memory": device.type == "cuda",
            "persistent_workers": NUM_WORKERS > 0,
        }
        if NUM_WORKERS > 0:
            loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
        train_loader = DataLoader(**loader_kwargs)
        log_print(f"Data loaded. Steps per epoch: {len(train_loader)}")
    except Exception as exc:
        log_print(f"Error loading dataset: {exc}")
        return

    model = FiveModalSleepModel(dim=EMB_DIM, seq_len=SEQ_LEN).to(device)
    if USE_CHANNELS_LAST and device.type == "cuda":
        for module in [model.cnn_eeg, model.cnn_eog, model.cnn_emg, model.cnn_ecg]:
            module.to(memory_format=torch.channels_last)

    if USE_TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)

    if is_distributed:
        if USE_SYNC_BN:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)

    map_location = _checkpoint_map_location(device)
    _load_checkpoint_if_available(model, PRETRAINED_P1_PATH, map_location, "P1")
    _load_checkpoint_if_available(model, PRETRAINED_P2_PATH, map_location, "P2")

    optimizer_kwargs = {"weight_decay": 1e-4}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True

    if not SKIP_PHASE1:
        log_print("\n>>> Phase 1: Hybrid Pairwise Alignment")
        opt1 = torch.optim.AdamW(model.parameters(), lr=LR_PHASE1, **optimizer_kwargs)
        for ep in range(1, EPOCHS_PHASE1 + 1):
            train_loader.sampler.set_epoch(ep)
            metrics = train_phase_1_loo_v2(model, train_loader, opt1, device, ep)
            log_msg = f"P1 Ep {ep}: " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            log_print(log_msg)
            if ep % 5 == 0 and is_main_process():
                torch.save(unwrap_model(model).state_dict(), f"{CHECKPOINT_DIR}/p1_ep{ep}.pth")
    else:
        log_print("\n>>> Phase 1 Skipped (Config: SKIP_PHASE1=True).")

    if not SKIP_PHASE2:
        log_print("\n>>> Phase 2: MAE Block-Masking Reconstruction + Contrastive")
        cnn_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "cnn" in name:
                cnn_params.append(param)
            else:
                other_params.append(param)

        opt2 = torch.optim.AdamW(
            [
                {"params": cnn_params, "lr": LR_PHASE2 * 0.1},
                {"params": other_params, "lr": LR_PHASE2},
            ],
            **optimizer_kwargs,
        )
        for ep in range(1, EPOCHS_PHASE2 + 1):
            train_loader.sampler.set_epoch(ep)
            metrics = train_phase_2_mae(
                model,
                train_loader,
                opt2,
                device,
                ep,
                pool_window=POOL_WINDOW,
                local_loss_weight=LOCAL_LOSS_WEIGHT,
            )
            log_msg = f"P2 Ep {ep}: " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            log_print(log_msg)
            if is_main_process():
                torch.save(unwrap_model(model).state_dict(), f"{CHECKPOINT_DIR}/p2_ep{ep}.pth")
    else:
        log_print("\n>>> Phase 2 Skipped (Config: SKIP_PHASE2=True).")

    log_print("\n>>> Training Finished.")
    if is_main_process():
        torch.save(unwrap_model(model).state_dict(), f"{CHECKPOINT_DIR}/final_model.pth")

    cleanup_distributed()


if __name__ == "__main__":
    main()
