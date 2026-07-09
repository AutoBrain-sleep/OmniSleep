"""Default configuration for OmniSleep pretraining."""

from __future__ import annotations

import os


CPU_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def apply_cpu_thread_defaults() -> None:
    """Set conservative CPU thread defaults unless the user already configured them."""
    for key, value in CPU_THREAD_ENV.items():
        os.environ.setdefault(key, value)


apply_cpu_thread_defaults()

TRAIN_LIST_PATH = "./dataset/splits/train_files.txt"
VAL_LIST_PATH = "./dataset/splits/val_files.txt"
CHECKPOINT_DIR = "./checkpoints/checkpoint_model_bert_block"

PRETRAINED_P1_PATH = "./checkpoints/pretrained/p1_ep30.pth"
PRETRAINED_P2_PATH = None

SKIP_PHASE1 = True
SKIP_PHASE2 = False

EPOCH_LEN_HIGH = 3000
EPOCH_LEN_LOW = 300
SEQ_LEN = 180
EMB_DIM = 256
PROJ_DIM = 128

BATCH_SIZE = 24
NUM_WORKERS = 12
PREFETCH_FACTOR = 4
SEED = 42

LR_PHASE1 = 8e-4
LR_PHASE2 = 5e-4

EPOCHS_PHASE1 = 50
EPOCHS_PHASE2 = 50

POOL_WINDOW = 60
LOCAL_LOSS_WEIGHT = 0.2

TEMPERATURE = 0.1
MAX_NEGATIVES = 1024
MAE_MASK_RATIO = 0.5

MASK_BLOCK_MIN = 5
MASK_BLOCK_MAX = 20

USE_TORCH_COMPILE = False
USE_CHANNELS_LAST = True
GRADIENT_ACCUMULATION_STEPS = 1
USE_SYNC_BN = False
