#!/usr/bin/env python
# coding: utf-8

# # Zero-Shot Image Retrieval Evaluation
# 
# This notebook demonstrates how we evaluate the trained `BrainAlignModel`.
# The core metric of BrainAlign is "Zero-Shot Top-K Retrieval". 
# 
# Given a brain recording (e.g. EEG response to an image), we pass it through our encoder to project it into the 512-dimensional CLIP space. We then compute the cosine similarity between this single brain vector and the *entire dictionary* of candidate 512D CLIP image representations in the test set.
# 
# If the true image the participant was viewing is the #1 most similar CLIP vector, that is a "Top-1" match. We also calculate Top-2 and Top-5.

# In[ ]:


import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

sys.path.append(os.path.abspath('..'))
from src.models.contrastive_model import BrainAlignModel
from src.data.eeg_loader import THINGSEEG2Dataset


# ### 1. Build the Candidate Test Set Cache. 
# 
# First, we load the `test` dataset split.

# In[ ]:


with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

clip_cache_path = os.path.join("..", config["data"]["clip_cache_dir"], "ViT-B-32.npz")
eeg_dir = os.path.join("..", config["data"]["eeg_dir"])

print("Loading Dataset...")
test_dataset = THINGSEEG2Dataset(eeg_dir=eeg_dir, clip_cache_path=clip_cache_path, split="test")
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Test Dataset contains {len(test_dataset)} trials across 200 unique test conditions.")


# In zero-shot retrieval, our candidates pool consists of the 200 unique test images. Let's build a matrix `(200, 512)` containing these ground-truth targets.

# In[ ]:


clip_dict = np.load(clip_cache_path)
unique_image_ids = set([Path(f).stem for f in test_dataset.files])

candidate_ids = list(unique_image_ids)
candidate_embeddings = []

for cid in candidate_ids:
    candidate_embeddings.append(clip_dict[cid])

# Shape: [200, 512]
candidate_matrix = torch.tensor(np.array(candidate_embeddings)).float()
# Normalize just to be safe (CLIP logic usually expects normalized vectors for cosine sim)
candidate_matrix = F.normalize(candidate_matrix, p=2, dim=-1)

print(f"Candidate Pool Matrix Shape: {candidate_matrix.shape}")


# ### 2. Scaffold the Model
# 
# Since we haven't completed a full 100-epoch training run yet, we will instantiate the model with random weights to scaffold the math logic. Random weights should produce ~0.5% (1/200) Top-1 accuracy.

# In[ ]:


sample = test_dataset[0]
in_channels, seq_len = sample["x"].shape

model = BrainAlignModel(
    in_channels=in_channels,
    seq_len=seq_len,
    brain_embed_dim=config["model"]["projection_dim"],
    clip_dim=512,
    tau_init=config["model"]["temperature_init"]
)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
model.eval()
candidate_matrix = candidate_matrix.to(device)
print(f"Model instantiated on {device}.")


# ### 3. Compute Retrieval Evaluation Loop
# We iterate through all trials in the test set. For each single trial's predicted embedding, we rank its cosine similarity against the 200 candidates.

# In[ ]:


top1 = 0
top2 = 0
top5 = 0
total = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating Retrieval"):
        x_brain = batch["x"].to(device)
        id_targets = batch["image_id"] # list of true ids

        # Get predicted 512D representations [Batch, 512]
        predictions = model(x_brain)
        predictions = F.normalize(predictions, p=2, dim=-1)

        # Cosine similarity matrix: [Batch, 200 candidates]
        # (Batch, 512) x (512, Candidates) -> (Batch, Candidates)
        sim_matrix = predictions @ candidate_matrix.T

        # Sort the candidates by similarity (descending)
        sorted_indices = torch.argsort(sim_matrix, dim=-1, descending=True)

        # Verify each item in batch
        for b_idx in range(len(id_targets)):
            true_id = id_targets[b_idx]

            # Get the candidate IDs of the top 5 predictions
            best_5_indices = sorted_indices[b_idx, :5].cpu().numpy()
            best_5_ids = [candidate_ids[i] for i in best_5_indices]

            if true_id == best_5_ids[0]: top1 += 1
            if true_id in best_5_ids[:2]: top2 += 1
            if true_id in best_5_ids: top5 += 1
            total += 1

print("\n--- Zero-Shot Retrieval Results ---")
print(f"Trials evaluated: {total}")
print(f"Top-1 Accuracy: {top1/total * 100:.2f}%")
print(f"Top-2 Accuracy: {top2/total * 100:.2f}%")
print(f"Top-5 Accuracy: {top5/total * 100:.2f}%")
print("Note: Since weights are random, expected accuracy over 200 classes is: Top-1 ~0.5%, Top-5 ~2.5%.")

