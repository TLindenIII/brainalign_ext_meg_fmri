from pathlib import Path
import os
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.eeg_loader import THINGSEEG2Dataset
from src.data.meg_loader import THINGSMEGDataset
from src.data.fmri_loader import THINGSfMRIDataset


with open(ROOT / "config.yaml", "r") as handle:
    config = yaml.safe_load(handle)

clip_cache_path = os.path.join(config["data"]["clip_cache_dir"], "ViT-B-32.npz")

print("--- EEG ---")
eeg = THINGSEEG2Dataset(config["data"]["eeg_dir"], clip_cache_path, split="train")
sample = eeg[0]
print(f"Size: {len(eeg)}, x: {sample['x'].shape}, y: {sample['y_clip'].shape}")

print("--- MEG ---")
meg = THINGSMEGDataset(config["data"]["meg_dir"], clip_cache_path)
if len(meg) > 0:
    sample = meg[0]
    print(f"Size: {len(meg)}, x: {sample['x'].shape}, y: {sample['y_clip'].shape}")

print("--- fMRI ---")
fmri = THINGSfMRIDataset(config["data"]["fmri_dir"], clip_cache_path)
if len(fmri) > 0:
    sample = fmri[0]
    print(f"Size: {len(fmri)}, x: {sample['x'].shape}, y: {sample['y_clip'].shape}")
