"""Dataset utilities for HDF5 sleep recordings listed in text files."""

from __future__ import annotations

from .config import EPOCH_LEN_HIGH, EPOCH_LEN_LOW

import bisect
import os
from collections import OrderedDict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class TextListDataset(Dataset):
    def __init__(self, txt_path, seq_len=20, stride=None, cache_size=32):
        super().__init__()
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.cache_size = cache_size

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"List file not found: {txt_path}")

        with open(txt_path, 'r') as f:
            self.files = [line.strip() for line in f.readlines() if line.strip()]

        self.file_paths = []
        self.cumulative_indices = []
        current_total = 0

        for f_path in self.files:
            try:
                with h5py.File(f_path, 'r') as f:
                    if 'y' in f:
                        total_epochs = f['y'].shape[0]
                        if total_epochs >= seq_len:
                            n_seq = (total_epochs - seq_len) // self.stride + 1
                            self.file_paths.append(f_path)
                            current_total += n_seq
                            self.cumulative_indices.append(current_total)
            except Exception:
                pass

        self.total_samples = current_total

        self.channel_map = {
            'eeg':  ['EEG'],
            'eog':  ['EOG'],
            'emg':  ['EMG'],
            'ecg':  ['ECG'],
            'resp': ['Resp_Airflow', 'Resp_Thorax', 'Resp_Abdomen']
        }

    def __len__(self):
        return self.total_samples

    def _get_dummy_output(self):
        dummy_sigs = {}
        dummy_masks = {}
        for k, keys in self.channel_map.items():
            t_len = EPOCH_LEN_LOW if k == 'resp' else EPOCH_LEN_HIGH
            dummy_sigs[k] = torch.zeros(len(keys), self.seq_len, t_len)
            dummy_masks[k] = 0.0
        return dummy_sigs, dummy_masks

    def __getitem__(self, idx):
        if not hasattr(self, 'file_handle_cache'):
            self.file_handle_cache = OrderedDict()

        file_idx = bisect.bisect_right(self.cumulative_indices, idx)
        if file_idx == 0:
            start_idx_global = 0
            selected_file = self.file_paths[0]
        else:
            start_idx_global = self.cumulative_indices[file_idx - 1]
            selected_file = self.file_paths[file_idx]

        local_seq_idx = idx - start_idx_global
        start_epoch = local_seq_idx * self.stride
        end_epoch = start_epoch + self.seq_len

        f = None
        try:
            if selected_file in self.file_handle_cache:
                f = self.file_handle_cache[selected_file]
                self.file_handle_cache.move_to_end(selected_file)
            else:
                f = h5py.File(selected_file, 'r', libver='latest', swmr=True)
                if len(self.file_handle_cache) >= self.cache_size:
                    old_path, old_f = self.file_handle_cache.popitem(last=False)
                    try: old_f.close()
                    except: pass
                self.file_handle_cache[selected_file] = f

            x_group = f['x']
            try:
                mask_grp_attrs = f['ch_presence_mask'].attrs
            except:
                mask_grp_attrs = {}

            sigs = {}
            masks = {}

            for mod_name, h5_keys in self.channel_map.items():
                is_present = True
                loaded_channels = []
                for key in h5_keys:
                    if key not in mask_grp_attrs or not mask_grp_attrs[key]:
                        if mask_grp_attrs:
                            is_present = False
                            break
                    data = x_group[key][start_epoch : end_epoch]
                    loaded_channels.append(data)

                if is_present and len(loaded_channels) == len(h5_keys):
                    data_np = np.stack(loaded_channels, axis=0)
                    sigs[mod_name] = torch.from_numpy(data_np).float()
                    masks[mod_name] = 1.0
                else:
                    t_len = EPOCH_LEN_LOW if mod_name == 'resp' else EPOCH_LEN_HIGH
                    sigs[mod_name] = torch.zeros(len(h5_keys), self.seq_len, t_len)
                    masks[mod_name] = 0.0

            return sigs, masks

        except Exception as e:
            if selected_file in self.file_handle_cache:
                del self.file_handle_cache[selected_file]
            try: f.close()
            except: pass
            return self._get_dummy_output()

    def __del__(self):
        if hasattr(self, 'file_handle_cache'):
            for f in self.file_handle_cache.values():
                try: f.close()
                except: pass
            self.file_handle_cache.clear()
