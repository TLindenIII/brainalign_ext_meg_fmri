from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.fmri_loader import THINGSfMRIDataset


with open(ROOT / "config.yaml", "r") as handle:
    config = yaml.safe_load(handle)

fmri = THINGSfMRIDataset(
    config["data"]["fmri_dir"],
    str(ROOT / "clip_cache" / "ViT-B-32.npz"),
)
if len(fmri) > 0:
    sample = fmri[0]
    print(f"Sample meta: {sample['meta']}")
