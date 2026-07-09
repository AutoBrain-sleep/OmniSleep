"""Model definitions for multimodal sleep representation learning."""

from __future__ import annotations

from .config import MASK_BLOCK_MAX, MASK_BLOCK_MIN

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.layer1 = self._make_layer(32, 64,  blocks=2, stride=2)
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
        self.roformers = nn.ModuleDict()      # Encoder
        self.mae_decoders = nn.ModuleDict()   # Decoder (4-layer Transformer)
        self.mae_projs = nn.ModuleDict()      # Reconstruction Head

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
