from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

meta = np.load(ROOT / "data" / "things-eeg2" / "stimuli" / "image_metadata.npy", allow_pickle=True).item()

print(list(meta.keys()))
print("Test image concepts array length:", len(meta["test_img_concepts_THINGS"]))
print("First five test concepts:", meta["test_img_concepts_THINGS"][:5])
print("First five train concepts:", meta["train_img_concepts_THINGS"][:5])
