# Linear Baseline Prototyping

To validate that the deep contrastive `BrainAlignModel` is actually learning complex, non-linear representations, we must compare its zero-shot retrieval performance against a classic linear baseline.

This notebook trains a simple **Ridge Regression** model from `scikit-learn` that attempts to map flattened brain matrices (e.g. EEG $17 \times 100$) directly to the 512D CLIP targets. We will evaluate its zero-shot retrieval accuracy.


```python
import os
import sys
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

sys.path.append(os.path.abspath('..'))
from src.data.eeg_loader import THINGSEEG2Dataset
```

### 1. Load the EEG Train/Test Splits


```python
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

clip_cache_path = os.path.join("..", config["data"]["clip_cache_dir"], "ViT-B-32.npz")
eeg_dir = os.path.join("..", config["data"]["eeg_dir"])

train_dataset = THINGSEEG2Dataset(eeg_dir=eeg_dir, clip_cache_path=clip_cache_path, split="train")
test_dataset = THINGSEEG2Dataset(eeg_dir=eeg_dir, clip_cache_path=clip_cache_path, split="test")

# Load the full training data into memory for Scikit-Learn 
# Note: EEG train is 66160 trials * 1700 floats (~450MB)
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=False)

X_train, Y_train = [], []
print("Extracting Training Matrices...")
for batch in tqdm(train_loader):
    # Flatten (Batch, 17, 100) -> (Batch, 1700)
    flat_x = batch['x'].view(batch['x'].size(0), -1).numpy()
    X_train.append(flat_x)
    Y_train.append(batch['y_clip'].numpy())

X_train = np.vstack(X_train)
Y_train = np.vstack(Y_train)

print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
```

### 2. Prepare the Candidate Dictionary


```python
from pathlib import Path

clip_dict = np.load(clip_cache_path)
unique_test_ids = list(set([Path(f).stem for f in test_dataset.files]))

test_candidates = []
for cid in unique_test_ids:
    test_candidates.append(clip_dict[cid])
    
test_candidates = np.array(test_candidates)
print(f"Scaffolded Candidate Evaluation Matrix: {test_candidates.shape}")
```

### 3. Train the Ridge Regressor


```python
print("Training Ridge Regression (alpha=100.0) \nThis may take a minute...")
ridge = Ridge(alpha=100.0)
ridge.fit(X_train, Y_train)
print("Training complete.")
```

### 4. Evaluate Top-1 / Top-5 Accuracy


```python
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

top1, top5, total = 0, 0, 0

print("Evaluating Test Set...")
for batch in tqdm(test_loader):
    # Inference flattened X
    X_test = batch['x'].view(batch['x'].size(0), -1).numpy()
    id_targets = batch['image_id']
    
    # Predict 512D targets
    Y_pred = ridge.predict(X_test)
    
    # Compute distances to candidates
    sims = cosine_similarity(Y_pred, test_candidates)
    sorted_indices = np.argsort(sims, axis=1)[:, ::-1]
    
    for i, true_id in enumerate(id_targets):
        best_5 = [unique_test_ids[idx] for idx in sorted_indices[i][:5]]
        if true_id == best_5[0]: top1 += 1
        if true_id in best_5: top5 += 1
        total += 1

print("\n--- Linear Baseline Results ---")
print(f"Trials: {total}")
print(f"Top-1 Retrieval: {top1/total * 100:.2f}%")
print(f"Top-5 Retrieval: {top5/total * 100:.2f}%")
```
