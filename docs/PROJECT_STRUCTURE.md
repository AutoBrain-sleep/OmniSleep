# Project Structure

```text
OmniSleep/
|-- configs/
|   `-- pretrain.yaml          # documented default experiment settings
|-- docs/
|   `-- PROJECT_STRUCTURE.md   # this file
|-- src/
|   `-- omnisleep/
|       |-- cli.py             # pretraining entry point
|       |-- config.py          # default paths and hyperparameters
|       |-- data.py            # HDF5 text-list dataset loader
|       |-- distributed.py     # DDP, seed, logging, metric reduction helpers
|       |-- losses.py          # contrastive objectives
|       |-- models.py          # CNN, RoPE transformer, and multimodal model
|       `-- training.py        # phase 1 and phase 2 training loops
|-- pretrain.py                # compatibility wrapper for `python pretrain.py`
|-- pyproject.toml             # package metadata
|-- requirements.txt           # runtime dependencies
|-- LICENSE
`-- README.md
```

Large datasets, generated checkpoints, logs, and model weights are intentionally
ignored by Git. Keep dataset split files under `dataset/splits/` locally and write
checkpoints under `checkpoints/`.
