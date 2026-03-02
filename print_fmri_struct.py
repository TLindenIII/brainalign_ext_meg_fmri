import yaml
from src.data.fmri_loader import THINGSfMRIDataset
with open('config.yaml', 'r') as f: config = yaml.safe_load(f)
fmri = THINGSfMRIDataset(config["data"]["fmri_dir"], "clip_cache/ViT-B-32.npz")
if len(fmri) > 0:
    s = fmri[0]
    print(f"Sample meta: {s['meta']}")
