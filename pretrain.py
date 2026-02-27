import os
# [CPU 核心优化]
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import datetime
import random
import sys
import math
import gc
import logging
import bisect 
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from collections import OrderedDict

# ================= 1. 配置区域 =================

TRAIN_LIST_PATH = "./dataset/splits/train_files.txt"
VAL_LIST_PATH = "./dataset/splits/val_files.txt"
CHECKPOINT_DIR = './checkpoints/checkpoint_model_bert_block'
PRETRAINED_P1_PATH = "./checkpoints/pretrained/p1_ep30.pth

# ----------------- [权重加载与阶段控制] -----------------

SKIP_PHASE1 = True  

# [Phase 2 配置]
PRETRAINED_P2_PATH = None 
SKIP_PHASE2 = False 

# ----------------------------------------------------------------

# --- 维度与模型参数 ---
EPOCH_LEN_HIGH = 3000 
EPOCH_LEN_LOW = 300   
SEQ_LEN = 180          
EMB_DIM = 256        
PROJ_DIM = 128        

# --- 训练超参数优化 ---
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
MAE_MASK_RATIO = 0.5  # 50% 掩码率

# [新增] 块状掩码配置
MASK_BLOCK_MIN = 5   # 最小连续掩码长度
MASK_BLOCK_MAX = 20  # 最大连续掩码长度

USE_TORCH_COMPILE = False 
USE_CHANNELS_LAST = True   
GRADIENT_ACCUMULATION_STEPS = 1 
USE_SYNC_BN = False 

# ================= 2. 分布式与日志工具 =================
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
        values_tensor = torch.tensor(values, device='cuda')
        dist.all_reduce(values_tensor)
        if average: values_tensor /= world_size
        return {k: v.item() for k, v in zip(keys, values_tensor)}

# ================= 3. 数据处理 =================

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
            'eeg':  ['EEG'],
            'eog':  ['EOG'],
            'emg':  ['EMG'],
            'ecg':  ['ECG'],
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

# ================= 4. 模型定义 =================
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size, stride, padding, dilation=dilation, groups=in_ch, bias=False),
            nn.BatchNorm1d(in_ch), nn.ReLU6(inplace=True))
        self.pointwise = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU6(inplace=True))
    def forward(self, x): return self.pointwise(self.depthwise(x))

class EnvelopeBranch(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=5, stride=stride, padding=2)
    def forward(self, x): return self.pool(torch.abs(x))

class MultiScaleEnvelopeStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert out_channels % 4 == 0
        branch_c = out_channels // 4 
        self.branch_high = DSConv(in_channels, branch_c, kernel_size=3, stride=2, padding=1)
        self.branch_mid = DSConv(in_channels, branch_c, kernel_size=7, stride=2, padding=3)
        self.branch_low = DSConv(in_channels, branch_c, kernel_size=7, stride=2, padding=12, dilation=4)
        self.branch_env_pre = EnvelopeBranch(stride=2)
        self.branch_env_post = nn.Sequential(nn.Conv1d(in_channels, branch_c, 1, bias=False), nn.BatchNorm1d(branch_c), nn.ReLU6(inplace=True))
        self.fusion = nn.Sequential(nn.Conv1d(branch_c * 4, out_channels, kernel_size=1, bias=False), nn.BatchNorm1d(out_channels), nn.ReLU6(inplace=True))
    def forward(self, x):
        h, m, l, e = self.branch_high(x), self.branch_mid(x), self.branch_low(x), self.branch_env_post(self.branch_env_pre(x))
        return self.fusion(torch.cat([h, m, l, e], dim=1))

class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel, expansion, activation, stride=1, padding=1):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channel * expansion
        self.conv1 = nn.Conv1d(in_channel, hidden_dim, kernel_size=1, bias=False)
        self.b0 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim, padding=padding, stride=stride, bias=False)
        self.b1 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, out_channel, kernel_size=1, stride=1, bias=False)
        self.b2 = nn.BatchNorm1d(out_channel)
        self.d = nn.Dropout(p=0.1) 
        self.act = activation(inplace=True) 
    def forward(self, x):
        use_res_connect = self.stride == 1 and x.shape[1] == self.conv3.out_channels
        y = self.act(self.b0(self.conv1(x)))
        y = self.act(self.b1(self.conv2(y)))
        y = self.b2(self.conv3(y))
        y = self.d(y)
        return x + y if use_res_connect else y

class MBConv(nn.Module):
    def __init__(self, in_channel, out_channels, expansion, layers, activation=nn.ReLU6, stride=2):
        super().__init__()
        self.stack = OrderedDict()
        self.stack['s0'] = Bottleneck(in_channel, out_channels, expansion, activation, stride=stride)
        for i in range(1, layers): self.stack['s'+str(i)] = Bottleneck(out_channels, out_channels, expansion, activation, stride=1)
        self.stack = nn.Sequential(self.stack)
    def forward(self, x): return self.stack(x)

class HybridEffNetEncoder(nn.Module):
    def __init__(self, in_channel=1, depth=[1, 1, 2, 2, 2, 2, 2], channels=[32, 24, 32, 48, 64, 96, 160, 256], expansion=3):
        super().__init__()
        self.stem = MultiScaleEnvelopeStem(in_channel, channels[0])
        self.b0 = nn.BatchNorm1d(channels[0]) 
        self.stages = nn.ModuleList()
        for i in range(len(depth)): self.stages.append(MBConv(channels[i], channels[i+1], expansion, depth[i], stride=2))
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = self.b0(self.stem(x))
        for stage in self.stages: x = stage(x)
        return self.pool(x).flatten(1)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    def forward(self, x):
        identity = x
        if self.downsample is not None: identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ECG_Encoder_ResNet(nn.Module):
    def __init__(self, output_dim=256):
        super(ECG_Encoder_ResNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False) 
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) 
        self.layer1 = self._make_layer(32, 64,  blocks=2, stride=2) 
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2) 
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2) 
        self.layer4 = self._make_layer(256, 256, blocks=2, stride=2) 
        self.avgpool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Linear(256, output_dim)
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks): layers.append(ResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) 
        x = self.avgpool(x) 
        x = torch.flatten(x, 1) 
        x = self.fc(x) 
        return x

class RespBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(RespBasicBlock, self).__init__()
        self.resp_conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.resp_bn1 = nn.BatchNorm1d(out_channels)
        self.resp_gelu = nn.GELU()
        self.resp_conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.resp_bn2 = nn.BatchNorm1d(out_channels)
        self.resp_downsample = None
        if stride != 1 or in_channels != out_channels:
            self.resp_downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    def forward(self, x):
        identity = x
        out = self.resp_conv1(x)
        out = self.resp_bn1(out)
        out = self.resp_gelu(out)
        out = self.resp_conv2(out)
        out = self.resp_bn2(out)
        if self.resp_downsample is not None: identity = self.resp_downsample(x)
        out += identity
        out = self.resp_gelu(out)
        return out

class RespBranchEncoder(nn.Module):
    def __init__(self):
        super(RespBranchEncoder, self).__init__()
        self.resp_stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.resp_layer1 = RespBasicBlock(32, 32, stride=1)
        self.resp_layer2 = RespBasicBlock(32, 64, stride=2)
        self.resp_layer3 = RespBasicBlock(64, 128, stride=2)
        self.resp_layer4 = RespBasicBlock(128, 256, stride=2)
        self.resp_avgpool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        x = self.resp_stem(x)
        x = self.resp_layer1(x)
        x = self.resp_layer2(x)
        x = self.resp_layer3(x)
        x = self.resp_layer4(x)
        x = self.resp_avgpool(x)
        return x.flatten(1)

class RespTriStreamModel(nn.Module):
    def __init__(self, output_dim=256):
        super(RespTriStreamModel, self).__init__()
        self.resp_flow_encoder = RespBranchEncoder()
        self.resp_thor_encoder = RespBranchEncoder()
        self.resp_abdo_encoder = RespBranchEncoder()
        self.resp_projector = nn.Sequential(
            nn.Linear(768, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim, bias=False)
        )
    def forward(self, x):
        x_flow = x[:, 0:1, :]
        x_thor = x[:, 1:2, :]
        x_abdo = x[:, 2:3, :]
        feat_flow = self.resp_flow_encoder(x_flow) 
        feat_thor = self.resp_thor_encoder(x_thor) 
        feat_abdo = self.resp_abdo_encoder(x_abdo) 
        resp_combined = torch.cat([feat_flow, feat_thor, feat_abdo], dim=1) 
        resp_out = self.resp_projector(resp_combined) 
        return resp_out
    
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d_model, max_seq_len=10000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
    def forward(self, x, seq_len=None):
        if seq_len is None: seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]

def apply_rotary_pos_emb(x, cos, sin):
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotate_half(x) * sin)

class RoPETransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbeddings(self.head_dim)
    def forward(self, src):
        B, N, C = src.shape
        x = self.norm1(src)
        q = self.q_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        cos_sin = self.rope(v, seq_len=N)
        cos, sin = cos_sin.cos(), cos_sin.sin()
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)
        src = src + self.dropout1(self.out_proj(attn_output))
        x = self.norm2(src)
        x = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        src = src + self.dropout2(x)
        return src

class FiveModalSleepModel(nn.Module):
    def __init__(self, dim=256, seq_len=180, proj_dim=128):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.proj_dim = proj_dim
        self.cnn_eeg = HybridEffNetEncoder(in_channel=1)
        self.cnn_eog = HybridEffNetEncoder(in_channel=1)
        self.cnn_emg = HybridEffNetEncoder(in_channel=1)
        self.cnn_ecg = ECG_Encoder_ResNet(output_dim=dim) 
        self.cnn_resp = RespTriStreamModel(output_dim=dim) 
        
        self.proj_eeg = self._build_proj(dim)
        self.proj_eog = self._build_proj(dim)
        self.proj_emg = self._build_proj(dim)
        self.proj_ecg = self._build_proj(dim)
        self.proj_resp = self._build_proj(dim)
        
        # Phase 2 Components: MAE (NO EMA)
        self.modalities = ['eeg', 'eog', 'emg', 'ecg', 'resp']
        self.roformers = nn.ModuleDict()      # Encoder
        self.mae_decoders = nn.ModuleDict()   # Decoder (4-layer Transformer)
        self.mae_projs = nn.ModuleDict()      # Reconstruction Head

        for m in self.modalities:
            # Encoder: 8 layers
            self.roformers[m] = nn.Sequential(*[RoPETransformerLayer(d_model=dim, nhead=8) for _ in range(8)], nn.LayerNorm(dim))
            
            # [MAE Decoder] 4-layer Transformer
            self.mae_decoders[m] = nn.Sequential(*[RoPETransformerLayer(d_model=dim, nhead=8) for _ in range(4)], nn.LayerNorm(dim))
            
            # Reconstruction Projector (Back to Feature Dim)
            self.mae_projs[m] = nn.Linear(dim, dim)

        # Tokens
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim))
        
    def _build_proj(self, dim):
        return nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, self.proj_dim))

    def _encode_all(self, inputs):
        B, _, L, _ = inputs['eeg'].shape
        def prep(x): return x.permute(0, 2, 1, 3).contiguous().view(B * L, x.shape[1], x.shape[3])
        z_eeg = self.cnn_eeg(prep(inputs['eeg'])) 
        z_eog = self.cnn_eog(prep(inputs['eog']))
        z_emg = self.cnn_emg(prep(inputs['emg']))
        z_ecg = self.cnn_ecg(prep(inputs['ecg']))
        z_resp = self.cnn_resp(prep(inputs['resp'])) 
        return z_eeg, z_eog, z_emg, z_ecg, z_resp

    def generate_block_mask(self, B, L, mask_ratio, device):
        """
        Generate block mask efficiently on GPU.
        0: Keep, 1: Mask
        """
        mask = torch.zeros((B, L), device=device, dtype=torch.bool)
        for b in range(B):
            num_masked = 0
            target_masked = int(L * mask_ratio)
            attempts = 0
            while num_masked < target_masked and attempts < 100:
                attempts += 1
                # Random block size between min and max
                block_size = torch.randint(MASK_BLOCK_MIN, MASK_BLOCK_MAX + 1, (1,)).item()
                if L - block_size <= 0: continue
                
                # Random start index
                start_idx = torch.randint(0, L - block_size, (1,)).item()
                
                # Check intersection (simplified: if mostly unmasked, mask it)
                if not torch.all(mask[b, start_idx : start_idx + block_size]):
                    mask[b, start_idx : start_idx + block_size] = True
                    num_masked = mask[b].sum().item()
        return mask.float()

    def block_masking(self, x, mask_ratio):
        """
        Perform Block Masking (Replace with Mask Token).
        x: [B, L, D]
        """
        B, L, D = x.shape
        
        # 1. Generate Binary Mask (0: Keep, 1: Mask)
        mask = self.generate_block_mask(B, L, mask_ratio, x.device) # [B, L]
        
        # 2. Expand mask for broadcasting
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, D) # [B, L, D]
        
        # 3. Replace masked positions with mask_token
        # mask_token: [1, 1, D] -> [B, L, D]
        mask_tokens = self.mask_token.expand(B, L, D)
        
        # x_masked = x * (1 - mask) + mask_token * mask
        x_masked = x * (1 - mask_expanded) + mask_tokens * mask_expanded
        
        return x_masked, mask

    def forward_phase1_v2(self, inputs, masks):
        z_eeg, z_eog, z_emg, z_ecg, z_resp = self._encode_all(inputs)
        p_eeg = F.normalize(self.proj_eeg(z_eeg), dim=1)
        p_eog = F.normalize(self.proj_eog(z_eog), dim=1)
        p_emg = F.normalize(self.proj_emg(z_emg), dim=1)
        p_ecg = F.normalize(self.proj_ecg(z_ecg), dim=1)
        p_resp = F.normalize(self.proj_resp(z_resp), dim=1)
        brain_pairs = [
            (p_eeg, F.normalize(p_eog + p_emg, dim=1)),
            (p_eog, F.normalize(p_eeg + p_emg, dim=1)),
            (p_emg, F.normalize(p_eeg + p_eog, dim=1))
        ]
        return brain_pairs, (p_ecg, p_resp), masks

    def forward_phase2_mae(self, inputs, mask_ratio=0.5, freeze_cnn=False, pool_window=5):
        # 1. Feature Encoding (Current CNN)
        if freeze_cnn:
            with torch.no_grad():
                z_eeg, z_eog, z_emg, z_ecg, z_resp = self._encode_all(inputs)
        else:
            z_eeg, z_eog, z_emg, z_ecg, z_resp = self._encode_all(inputs)
        
        B, _, L, _ = inputs['eeg'].shape
        
        feats_map = {
            'eeg': z_eeg.view(B, L, self.dim),
            'eog': z_eog.view(B, L, self.dim),
            'emg': z_emg.view(B, L, self.dim),
            'ecg': z_ecg.view(B, L, self.dim),
            'resp': z_resp.view(B, L, self.dim)
        }
        
        outputs = {'recon': {}, 'targets': {}, 'mask': {}, 'latent_full': {}}
        
        for m in self.modalities:
            x = feats_map[m] # [B, L, D]
            
            # --- MAE Target Setup ---
            target = x.clone().detach() 
            
            # --- MAE Process (Block Masking) ---
            # 1. Masking (Replace mode)
            x_masked, mask = self.block_masking(x, mask_ratio)
            
            # 2. Encoder (Full Sequence)
            # Input already has mask tokens at masked positions
            latent = self.roformers[m](x_masked)
            
            # 3. Decoder (Full Sequence)
            dec_out = self.mae_decoders[m](latent)
            
            # 4. Projection to Feature Dim
            pred = self.mae_projs[m](dec_out)
            
            outputs['recon'][m] = pred
            outputs['targets'][m] = target
            outputs['mask'][m] = mask
            outputs['latent_full'][m] = latent

        # --- Contrastive Logic (Pairwise) ---
        flat_eeg = outputs['latent_full']['eeg'].reshape(B*L, -1)
        flat_eog = outputs['latent_full']['eog'].reshape(B*L, -1)
        flat_emg = outputs['latent_full']['emg'].reshape(B*L, -1)
        flat_ecg = outputs['latent_full']['ecg'].reshape(B*L, -1)
        flat_resp = outputs['latent_full']['resp'].reshape(B*L, -1)
        
        p_eeg = F.normalize(self.proj_eeg(flat_eeg), dim=1)
        p_eog = F.normalize(self.proj_eog(flat_eog), dim=1)
        p_emg = F.normalize(self.proj_emg(flat_emg), dim=1)
        p_ecg = F.normalize(self.proj_ecg(flat_ecg), dim=1)
        p_resp = F.normalize(self.proj_resp(flat_resp), dim=1)
        
        # Local Internal (Pairwise)
        brain_pairs = [
            (p_eeg, p_eog),
            (p_eeg, p_emg),
            (p_eog, p_emg)
        ]
        body_pair_local = (p_ecg, p_resp)

        # Cross System (Pairwise Pooled)
        num_windows = L // pool_window
        def pool_and_norm(tensor_flat):
            pooled = tensor_flat.view(B, num_windows, pool_window, -1).mean(dim=2)
            return F.normalize(pooled.reshape(-1, self.proj_dim), dim=1)
            
        p_eeg_pool = pool_and_norm(p_eeg)
        p_eog_pool = pool_and_norm(p_eog)
        p_emg_pool = pool_and_norm(p_emg)
        p_ecg_pool = pool_and_norm(p_ecg)
        p_resp_pool = pool_and_norm(p_resp)
        
        cross_pairs = [
            (p_eeg_pool, p_resp_pool),
            (p_eeg_pool, p_ecg_pool),
            (p_eog_pool, p_resp_pool),
            (p_eog_pool, p_ecg_pool),
            (p_emg_pool, p_resp_pool),
            (p_emg_pool, p_ecg_pool)
        ]
        
        return outputs, cross_pairs, (brain_pairs, body_pair_local)

# ================= 5. 训练循环优化 =================

def train_phase_1_loo_v2(model, loader, optimizer, device, epoch):
    model.train()
    acc_loss = torch.tensor(0.0, device=device)
    acc_brain = torch.tensor(0.0, device=device)
    acc_body = torch.tensor(0.0, device=device)
    steps = 0
    if is_main_process(): 
        pbar = tqdm(loader, desc=f"P1 (Hybrid) Ep {epoch}", ncols=120)
    else: 
        pbar = loader
    for step, (inputs, masks) in enumerate(pbar):
        inputs = {k: v.to(device, non_blocking=True) for k,v in inputs.items()}
        masks = {k: v.to(device, non_blocking=True) for k,v in masks.items()}
        B, _, L, _ = inputs['eeg'].shape
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            brain_pairs, body_pair, _ = model.module.forward_phase1_v2(inputs, masks)
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
        if is_main_process() and step % 20 == 0:
            pbar.set_postfix({
                'L': f"{loss.item():.4f}", 
                'Br': f"{l_brain.item():.4f}", 
                'Bo': f"{l_body.item():.4f}"
            })
    results = {'Loss': acc_loss / steps, 'BrainLoss': acc_brain / steps, 'BodyLoss': acc_body / steps}
    return reduce_dict(results)

def train_phase_2_mae(model, loader, optimizer, device, epoch, pool_window=5, local_loss_weight=0.5):
    model.train()
    
    acc_loss = torch.tensor(0.0, device=device)
    acc_mae = torch.tensor(0.0, device=device)
    acc_cross = torch.tensor(0.0, device=device)
    acc_local = torch.tensor(0.0, device=device) 
    steps = 0
    
    if is_main_process(): 
        pbar = tqdm(loader, desc=f"P2 (MAE) Ep {epoch}", ncols=120)
    else: 
        pbar = loader
        
    for step, (inputs, masks) in enumerate(pbar):
        inputs = {k: v.to(device, non_blocking=True) for k,v in inputs.items()}
        masks = {k: v.to(device, non_blocking=True) for k,v in masks.items()}
        B, _, L, _ = inputs['eeg'].shape
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Unfreeze CNN: freeze_cnn=False
            outputs, cross_pairs, (brain_pairs, body_pair_local) = model.module.forward_phase2_mae(
                inputs, mask_ratio=MAE_MASK_RATIO, freeze_cnn=False, pool_window=pool_window
            )
            
            # 1. MAE Reconstruction Loss
            loss_mae = 0
            for m in model.module.modalities:
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

            loss_mae /= len(model.module.modalities)
            
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
            
        if is_main_process() and step % 20 == 0:
            pbar.set_postfix({
                'L': f"{loss.item():.4f}",
                'MAE': f"{loss_mae.item():.4f}",
                'C': f"{loss_cross.item():.4f}",
                'Loc': f"{loss_local.item():.4f}"
            })
            if logger:
                for handler in logger.handlers:
                    handler.flush()
            
    results = {
        'Loss': acc_loss/steps, 
        'MAELoss': acc_mae/steps, 
        'CrossLoss': acc_cross/steps,
        'LocalLoss': acc_local/steps
    }
    return reduce_dict(results)

# ================= 6. 主程序 =================

def main():
    gc.collect()
    torch.cuda.empty_cache()
    
    is_distributed, rank, world_size, gpu = init_distributed_mode()
    setup_logger(CHECKPOINT_DIR, rank)
    device = torch.device(f"cuda:{gpu}")
    log_print(f"Batch Size: {BATCH_SIZE} | Workers: {NUM_WORKERS}")

    try:
        train_dataset = TextListDataset(TRAIN_LIST_PATH, seq_len=SEQ_LEN)
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            sampler=train_sampler, 
            num_workers=NUM_WORKERS, 
            drop_last=True, 
            pin_memory=True, 
            persistent_workers=True, 
            prefetch_factor=PREFETCH_FACTOR
        )
        
        log_print(f"Data loaded. Steps per epoch: {len(train_loader)}")
        
    except Exception as e:
        log_print(f"Error loading dataset: {e}")
        return

    # 初始化模型
    model = FiveModalSleepModel(dim=EMB_DIM, seq_len=SEQ_LEN).to(device)
    if USE_CHANNELS_LAST:
        for m in [model.cnn_eeg, model.cnn_eog, model.cnn_emg, model.cnn_ecg]: 
            m.to(memory_format=torch.channels_last)
            
    if is_distributed:
        if USE_SYNC_BN:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)
    
    # -------------------------------------------------------------
    # 权重加载逻辑
    # -------------------------------------------------------------
    map_location = torch.device(f"cuda:{gpu}")
    
    if PRETRAINED_P1_PATH and os.path.exists(PRETRAINED_P1_PATH):
        log_print(f"\n[Config] Loading Pretrained P1 Weights: {PRETRAINED_P1_PATH}")
        checkpoint = torch.load(PRETRAINED_P1_PATH, map_location=map_location)
        model.module.load_state_dict(checkpoint, strict=False)
        log_print("-> P1 Weights Loaded.")
    elif PRETRAINED_P1_PATH:
        log_print(f"\n[Warning] P1 Path not found: {PRETRAINED_P1_PATH}")

    if PRETRAINED_P2_PATH and os.path.exists(PRETRAINED_P2_PATH):
        log_print(f"\n[Config] Loading Pretrained P2 Weights: {PRETRAINED_P2_PATH}")
        checkpoint = torch.load(PRETRAINED_P2_PATH, map_location=map_location)
        model.module.load_state_dict(checkpoint, strict=False)
        log_print("-> P2 Weights Loaded.")
    elif PRETRAINED_P2_PATH:
        log_print(f"\n[Warning] P2 Path not found: {PRETRAINED_P2_PATH}")
    
    
    # -------------------------------------------------------------
    # [Phase 1 执行逻辑]
    # -------------------------------------------------------------
    if not SKIP_PHASE1:
        log_print("\n>>> Phase 1: Hybrid Pairwise Alignment")
        opt1 = torch.optim.AdamW(model.parameters(), lr=LR_PHASE1, weight_decay=1e-4, fused=True)
        for ep in range(1, EPOCHS_PHASE1 + 1):
            train_loader.sampler.set_epoch(ep) 
            metrics = train_phase_1_loo_v2(model, train_loader, opt1, device, ep)
            
            # [LOGGING] Log all loss components for Phase 1
            log_msg = f"P1 Ep {ep}: " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            log_print(log_msg)
            
            if ep % 5 == 0 and is_main_process(): 
                torch.save(model.module.state_dict(), f"{CHECKPOINT_DIR}/p1_ep{ep}.pth")
    else:
        log_print("\n>>> Phase 1 Skipped (Config: SKIP_PHASE1=True).")

    # -------------------------------------------------------------
    # [Phase 2 执行逻辑] - MAE Mode (No EMA)
    # -------------------------------------------------------------
    if not SKIP_PHASE2:
        log_print("\n>>> Phase 2: MAE Block-Masking Reconstruction + Contrastive")
        
        # Differential Learning Rates
        cnn_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if 'cnn' in name:
                cnn_params.append(param)
            else:
                other_params.append(param)
        
        opt2 = torch.optim.AdamW([
            {'params': cnn_params, 'lr': LR_PHASE2 * 0.1}, 
            {'params': other_params, 'lr': LR_PHASE2}
        ], weight_decay=1e-4, fused=True)
        
        for ep in range(1, EPOCHS_PHASE2 + 1):
            train_loader.sampler.set_epoch(ep)
            metrics = train_phase_2_mae(model, train_loader, opt2, device, ep, pool_window=POOL_WINDOW, local_loss_weight=LOCAL_LOSS_WEIGHT)
            
            # [LOGGING] Log all loss components for Phase 2
            log_msg = f"P2 Ep {ep}: " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            log_print(log_msg)
            
            if ep % 1 == 0 and is_main_process(): 
                torch.save(model.module.state_dict(), f"{CHECKPOINT_DIR}/p2_ep{ep}.pth")
    else:
        log_print("\n>>> Phase 2 Skipped (Config: SKIP_PHASE2=True).")

    log_print("\n>>> Training Finished.")

    if is_main_process(): 
        torch.save(model.module.state_dict(), f"{CHECKPOINT_DIR}/final_model.pth")
    
    cleanup_distributed()

if __name__ == "__main__":
    main()

