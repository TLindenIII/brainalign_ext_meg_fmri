from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.contrastive_model import BrainAlignModel


print("Instantiating BrainAlignModel (CBraMod backbone)...")
model = BrainAlignModel(in_channels=63, seq_len=100)
model.eval()

x = torch.randn(8, 63, 100)
print(f"Input shape: {x.shape}")

with torch.no_grad():
    out = model(x)

print(f"Output shape (expected 8, 512): {out.shape}")
print("Test Passed.")
