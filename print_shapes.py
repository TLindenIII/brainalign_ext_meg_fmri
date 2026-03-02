import os
import sys
import yaml
from src.data.eeg_loader import THINGSEEG2Dataset
from src.data.meg_loader import THINGSMEGDataset
from src.data.fmri_loader import THINGSfMRIDataset

with open('config.yaml', 'r') as f: config = yaml.safe_load(f)
clip_cache_path = os.path.join(config["data"]["clip_cache_dir"], "ViT-B-32.npz")

print("--- EEG ---")
eeg = THINGSEEG2Dataset(config["data"]["eeg_dir"], clip_cache_path, split="train")
s = eeg[0]
print(f"Size: {len(eeg)}, x: {s['x'].shape}, y: {s['y_clip'].shape}")

print("--- MEG ---")
meg = THINGSMEGDataset(config["data"]["meg_dir"], clip_cache_path)
if len(meg) > 0:
    s = meg[0]
    print(f"Size: {len(meg)}, x: {s['x'].shape}, y: {s['y_clip'].shape}")

print("--- fMRI ---")
fmri = THINGSfMRIDataset(config["data"]["fmri_dir"], clip_cache_path)
if len(fmri) > 0:
    s = fmri[0]
    print(f"Size: {len(fmri)}, x: {s['x'].shape}, y: {s['y_clip'].shape}")
