import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _TemporalResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, dilation=1, dropout=0.2):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.block(x)
        x = self.dropout(x)
        return F.gelu(x + residual)


class MEGAlignModel(nn.Module):
    """
    MEG-specific encoder that preserves the full sensor layout instead of
    forcing 271 sensors through an EEG-shaped 63-channel CBraMod adapter.
    """

    def __init__(
        self,
        in_channels,
        seq_len,
        clip_dim=512,
        hidden_dim=256,
        dropout=0.2,
        tau_init=0.07,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len

        self.sensor_projection = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.temporal_stem = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=9,
                padding=4,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.temporal_blocks = nn.ModuleList(
            [
                _TemporalResidualBlock(hidden_dim, kernel_size=7, dilation=1, dropout=dropout),
                _TemporalResidualBlock(hidden_dim, kernel_size=7, dilation=2, dropout=dropout),
                _TemporalResidualBlock(hidden_dim, kernel_size=7, dilation=4, dropout=dropout),
                _TemporalResidualBlock(hidden_dim, kernel_size=7, dilation=8, dropout=dropout),
            ]
        )

        self.time_attention = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=1),
        )

        self.feature_norm = nn.LayerNorm(hidden_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, clip_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tau_init))

    def forward(self, x_brain):
        if x_brain.dim() == 4 and x_brain.size(2) == 1:
            x_brain = x_brain.squeeze(2)
        if x_brain.dim() == 2:
            x_brain = x_brain.unsqueeze(1)

        x = self.sensor_projection(x_brain)
        x = self.temporal_stem(x)
        for block in self.temporal_blocks:
            x = block(x)

        time_weights = torch.softmax(self.time_attention(x).squeeze(1), dim=-1)
        pooled = torch.sum(x * time_weights.unsqueeze(1), dim=-1)
        pooled = self.feature_norm(pooled)

        p_brain = self.projection_head(pooled)
        return F.normalize(p_brain, dim=-1, eps=1e-6)
