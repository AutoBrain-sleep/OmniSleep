"""Distributed training and logging helpers."""

from __future__ import annotations

from .config import SEED

import datetime
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist

def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()
    setup_seed(SEED + rank)
    print(f"Distributed init: Rank {rank}, GPU {gpu}, World Size {world_size}")
    return True, rank, world_size, gpu

def is_main_process():
    return dist.get_rank() == 0 if dist.is_initialized() else True

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

logger = None
def setup_logger(save_dir, rank):
    if rank != 0: return None
    os.makedirs(save_dir, exist_ok=True)
    global logger
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(save_dir, f'train_log_{current_time}.txt')
    logger = logging.getLogger("TrainLogger")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger

def log_print(msg):
    if is_main_process():
        if logger: logger.info(msg)
        else: print(f"[{datetime.datetime.now()}] {msg}", flush=True)

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def reduce_dict(input_dict, average=True):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size < 2:
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}
    with torch.no_grad():
        keys = sorted(input_dict.keys())
        values = [input_dict[k] for k in keys]
        device = next((v.device for v in values if isinstance(v, torch.Tensor)), torch.device("cuda"))
        values_tensor = torch.stack([
            v.detach() if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)
            for v in values
        ]).to(device)
        dist.all_reduce(values_tensor)
        if average: values_tensor /= world_size
        return {k: v.item() for k, v in zip(keys, values_tensor)}
