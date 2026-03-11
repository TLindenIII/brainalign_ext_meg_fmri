import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add vendored CBraMod repository to the *front* of the path 
# so its internal 'from models.criss_cross_transformer' import 
# resolves to src/vendored/CBraMod/models instead of our own src/models.
vendored_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vendored', 'CBraMod'))
if vendored_path not in sys.path:
    sys.path.insert(0, vendored_path)

from models.cbramod import CBraMod

class BrainAlignModel(nn.Module):
    """
    The full Contrastive Alignment Model:
    1. Brain Encoder (Official CBraMod / Criss-Cross Transformer)
    2. Projection Head (MLP mapping to CLIP dim)
    3. Learnable Temperature
    """
    
    def __init__(self, in_channels=63, seq_len=100, brain_embed_dim=512, clip_dim=512, tau_init=0.07):
        super().__init__()
        
        # Spatial channel alignment: mapping arbitrary channels to 63 (expected by pretrained CBraMod)
        self.in_channels = in_channels
        self.spatial_mapping = nn.Conv1d(in_channels, 63, kernel_size=1) if in_channels != 63 else None
        
        # NOTE: 
        # The THINGS-EEG2 dataset has a sequence length of 100 per trial (100ms at 1000Hz).
        # CBraMod's patch encoder mathematically expects inputs structured into patches.
        # To avoid mismatching the complex positional embeddings, we pass it forward as 
        # a single patch of dimension 100: shape (B, C, 1, 100).
        self.patch_size = seq_len
        self.num_patches = 1
        
        # Core Brain Encoder
        # The pre-trained CBraMod architecture strictly assumes `in_dim=200` due to its internal 
        # rfft spectral projections (101 bins) and 1D patch convolutions. 
        # We must therefore fix our patch length to 200, regardless of the input seq_len.
        self.patch_size = 200
        
        self.brain_encoder = CBraMod(
            in_dim=self.patch_size, 
            out_dim=200, 
            d_model=200, 
            dim_feedforward=800, 
            seq_len=self.num_patches, # Note: cbramod patch_embedding uses this seq_len
            n_layer=12,         
            nhead=8
        )
        
        # Discover and load HuggingFace pre-trained weights if downloaded
        weights_path = os.path.join(vendored_path, "pretrained_weights", "pretrained_weights.pth")
        if os.path.exists(weights_path):
            print(f"Loading pretrained CBraMod weights from: {weights_path}")
            self.brain_encoder.load_state_dict(torch.load(weights_path, map_location='cpu'))
        else:
            print("Warning: Pretrained weights not found. Training CBraMod from scratch!")
        
        # If the incoming seq_len != 200, we will interpolate it in the forward pass.
        self.seq_len = seq_len
        
        # Projection Head (as defined in BrainAlign: Linear -> ReLU -> Linear)
        self.projection_head = nn.Sequential(
            nn.Linear(200, brain_embed_dim),
            nn.ReLU(),
            nn.Linear(brain_embed_dim, clip_dim)
        )
        
        # Learnable temperature scalar (initialized to CLIP's logit scale)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / tau_init))
        
    def forward(self, x_brain):
        # The input x_brain from datasets like THINGSEEG2Dataset is (B, Channels, Time) -> (B, 63, 100)
        
        # 0. Spatial mapping if input channels don't match 63
        if self.spatial_mapping is not None:
            x_brain = self.spatial_mapping(x_brain)

        # 1. Resample Time axis to exactly 200 to satisfy CBraMod's patch requirements
        if x_brain.size(-1) != self.patch_size:
            x_brain = F.interpolate(x_brain, size=self.patch_size, mode='linear', align_corners=False)
            
        # 2. Reshape to CBraMod's expected (B, Channels, Patches, PatchSize)
        # We model the entire trial as 1 Patch. -> (B, 63, 1, 200)
        if x_brain.dim() == 3:
            x_brain = x_brain.unsqueeze(2) 
            
        # Extract brain representation
        # CBraMod outputs shape (B, Channels, Patches, OutDim) -> (B, 63, 1, 512)
        z_brain = self.brain_encoder(x_brain)
        
        # We flatten spatial/patch dimensions down to (B, OutDim)
        # BrainAlign uses average pooling across the channels/patches before projecting
        z_brain = z_brain.mean(dim=(1, 2)) # Shape: (B, 512)
        
        # Project to CLIP space
        p_brain = self.projection_head(z_brain)
        
        # Normalize to unit sphere (essential for cosine similarity & InfoNCE)
        p_brain = p_brain / p_brain.norm(dim=-1, keepdim=True)
        
        return p_brain
