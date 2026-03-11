import numpy as np
import torch
import torch.nn as nn


class fMRIAlignModel(nn.Module):
    """
    Contrastive alignment model for fMRI voxel data → CLIP space.
    
    Designed for variance-selected voxel inputs (~15K dims). Uses a
    bottleneck MLP with residual connections, LayerNorm, GELU activations,
    and dropout for strong regularization.
    
    Architecture:
        voxels (15K) → Linear(V, 2048) → ResBlock(2048) → Linear(2048, 512) → L2-norm
    
    With 15K input: ~31M params (vs 934M with full 211K voxels)
    """
    
    def __init__(self, n_voxels, clip_dim=512, hidden_dim=2048, dropout=0.5, tau_init=0.07):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_voxels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Residual refinement block
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Output projection to CLIP space
        self.output_proj = nn.Linear(hidden_dim, clip_dim)
        
        # Learnable temperature scalar (clamped in forward pass)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tau_init))
        
    def forward(self, x_brain):
        """
        Args:
            x_brain: (B, 1, V) or (B, V) — globally z-scored voxel vector
        Returns:
            p_brain: (B, clip_dim) — L2-normalized projection
        """
        if x_brain.dim() == 3:
            x_brain = x_brain.squeeze(1)
        
        # Clamp logit_scale (OpenAI CLIP uses max=ln(100))
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=np.log(100))
            
        h = self.input_proj(x_brain)
        h = h + self.res_block(h)  # Residual connection
        p_brain = self.output_proj(h)
        
        # L2 normalize to unit sphere
        p_brain = p_brain / p_brain.norm(dim=-1, keepdim=True)
        
        return p_brain
