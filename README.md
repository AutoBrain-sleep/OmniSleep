# OmniSleep

OmniSleep is a multimodal sleep representation pretraining project for PSG signals, including EEG, EOG, EMG, ECG, and respiratory channels. The training pipeline combines leave-one-out contrastive alignment with masked autoencoding and cross-system contrastive learning.

<img width="3644" height="905" alt="OmniSleep overview" src="https://github.com/user-attachments/assets/617ad75a-0b86-4edf-9d26-bb7aa627de63" />

<img width="1641" height="1118" alt="OmniSleep model details" src="https://github.com/user-attachments/assets/03694247-7541-48d7-8bbd-9e0ad7abc28b" />

<img width="1912" height="384" alt="OmniSleep training stages" src="https://github.com/user-attachments/assets/bec0deb3-2f46-43a8-aa02-4759913278b5" />

## Repository Layout

The original single-file script has been organized into a Python package:

- `src/omnisleep/config.py`: default paths, model dimensions, and training hyperparameters
- `src/omnisleep/data.py`: HDF5 text-list dataset loader
- `src/omnisleep/models.py`: multimodal encoders, RoPE transformer blocks, and `FiveModalSleepModel`
- `src/omnisleep/losses.py`: contrastive learning objectives
- `src/omnisleep/training.py`: Phase 1 and Phase 2 training loops
- `src/omnisleep/cli.py`: command-line training entry point
- `configs/pretrain.yaml`: documented experiment defaults

See `docs/PROJECT_STRUCTURE.md` for the full tree.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

Install a CUDA-enabled PyTorch build that matches your GPU and driver when training on GPU.

## Data Layout

By default, training reads file lists from:

```text
dataset/splits/train_files.txt
dataset/splits/val_files.txt
```

Each line in `train_files.txt` should point to an HDF5 file. The loader expects the following groups/keys:

- `x/EEG`
- `x/EOG`
- `x/EMG`
- `x/ECG`
- `x/Resp_Airflow`
- `x/Resp_Thorax`
- `x/Resp_Abdomen`
- `y` for epoch count
- optional `ch_presence_mask` attributes for channel availability

## Training

Run with the compatibility wrapper:

```bash
python pretrain.py
```

Or, after installation:

```bash
python -m omnisleep.cli
```

Multi-GPU distributed training can be launched with `torchrun`:

```bash
torchrun --nproc_per_node=4 -m omnisleep.cli
```

Default hyperparameters live in `src/omnisleep/config.py`; `configs/pretrain.yaml` mirrors them for experiment tracking.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
