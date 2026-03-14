from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


weights = torch.load(
    ROOT / "src" / "vendored" / "CBraMod" / "pretrained_weights" / "pretrained_weights.pth",
    map_location="cpu",
)
for key, value in weights.items():
    if len(value.shape) > 0:
        print(f"{key}: {value.shape}")
    else:
        print(f"{key}: scalar")
