import os
import sys
from pathlib import Path
import numpy as np
import yaml
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cross_decomposition import CCA
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.abspath('..'))

from src.data.eeg_loader import THINGSEEG2Dataset
from src.data.meg_loader import THINGSMEGDataset
from src.data.fmri_loader import THINGSfMRIDataset

print("Loading config...")
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

clip_cache_path = Path(config["data"]["clip_cache_dir"]) / "ViT-B-32.npz"

def load_data(modality, split):
    print(f"Loading {modality.upper()} {split} data...")
    if modality == "eeg":
        dataset = THINGSEEG2Dataset(
            eeg_dir=Path(config["data"]["eeg_dir"]),
            clip_cache_path=str(clip_cache_path),
            split=split
        )
    elif modality == "meg":
        dataset = THINGSMEGDataset(
            meg_dir=Path(config["data"]["meg_dir"]),
            clip_cache_path=str(clip_cache_path),
            split=split
        )
    elif modality == "fmri":
        dataset = THINGSfMRIDataset(
            fmri_dir=Path(config["data"]["fmri_dir"]),
            clip_cache_path=str(clip_cache_path),
            split=split
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")
        
    X, Y, ids = [], [], []
    for i in tqdm(range(len(dataset)), desc=f"Parsing {modality.upper()} {split}"):
        item = dataset[i]
        # Flatten brain features to 1D vector per trial for linear regression
        X.append(item["x"].numpy().flatten())
        Y.append(item["y_clip"].numpy())
        ids.append(item["image_id"])
        
    return np.array(X), np.array(Y), ids

def evaluate_retrieval(Y_pred, Y_true, true_ids, clip_dict):
    """
    Evaluates Top-1 and Top-5 retrieval accuracy.
    Matches the eval scheme: per-image averaged embeddings vs all candidates.
    """
    unique_test_ids = list(set(true_ids))
    
    test_candidates = []
    for cid in unique_test_ids:
        test_candidates.append(clip_dict[cid])
    test_candidates = np.array(test_candidates)
    
    target_p_brains = {tid: [] for tid in unique_test_ids}
    
    for i, true_id in enumerate(true_ids):
        target_p_brains[true_id].append(Y_pred[i])
        
    top1, top5, total = 0, 0, 0
    
    for true_id, p_brain_trials in target_p_brains.items():
        if len(p_brain_trials) == 0: continue
            
        avg_p_brain = np.mean(p_brain_trials, axis=0, keepdims=True)
        
        sims = cosine_similarity(avg_p_brain, test_candidates)
        sorted_indices = np.argsort(sims, axis=1)[:, ::-1]
        
        best_5 = [unique_test_ids[idx] for idx in sorted_indices[0][:5]]
        if true_id == best_5[0]: top1 += 1
        if true_id in best_5: top5 += 1
        total += 1
                
    t1_acc = (top1/total)*100 if total > 0 else 0
    t5_acc = (top5/total)*100 if total > 0 else 0
    return t1_acc, t5_acc

def run_baselines(modality):
    print(f"\n{'='*40}\nRunning Linear Baselines for {modality.upper()}\n{'='*40}")
    
    X_train, Y_train, _ = load_data(modality, "train")
    X_test, Y_test, test_ids = load_data(modality, "test")
    
    print(f"\nTraining Models...")
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    
    # 1. Ridge Regression
    print("Fitting Ridge Regression...")
    ridge = Ridge(alpha=10000.0)
    ridge.fit(X_train, Y_train)
    
    print("Predicting Ridge test set...")
    Y_pred_ridge = ridge.predict(X_test)
    
    # 2. Canonical Correlation Analysis (CCA)
    # Project both Brain and CLIP features into a shared 512-dim subspace
    print("Fitting CCA (Canonical Correlation Analysis)...")
    cca = CCA(n_components=100) # Keep components reasonable for sklearn memory
    cca.fit(X_train, Y_train)
    
    print("Predicting CCA test set...")
    # CCA transform yields (X_c, Y_c). We use X_c as the "brain embedding" in the shared space
    X_test_c, Y_test_c = cca.transform(X_test, Y_test)
    
    # To do retrieval we need the candidate CLIP embeddings projected into the CCA space too!
    clip_dict = np.load(clip_cache_path)
    cca_clip_dict = {}
    
    # We must transform the entire candidate vocabulary into the CCA space
    unique_test_ids = list(set(test_ids))
    vocab = np.array([clip_dict[cid] for cid in unique_test_ids])
    _, vocab_c = cca.transform(np.zeros((len(vocab), X_train.shape[1])), vocab)
    
    for i, cid in enumerate(unique_test_ids):
        cca_clip_dict[cid] = vocab_c[i]
    
    # Evaluate
    print("\nScoring...")
    t1_ridge, t5_ridge = evaluate_retrieval(Y_pred_ridge, Y_test, test_ids, clip_dict)
    t1_cca, t5_cca = evaluate_retrieval(X_test_c, Y_test_c, test_ids, cca_clip_dict) # X_c retrieves from Y_c candidates
    
    print(f"\n--- Results ({modality.upper()}) ---")
    print(f"[Ridge -> CLIP] Top-1: {t1_ridge:.2f}% | Top-5: {t5_ridge:.2f}%")
    print(f"[CCA Shared Space] Top-1: {t1_cca:.2f}% | Top-5: {t5_cca:.2f}%")
    
    return {"ridge_t1": t1_ridge, "ridge_t5": t5_ridge, "cca_t1": t1_cca, "cca_t5": t5_cca}

if __name__ == "__main__":
    results = {}
    for mod in ["eeg", "meg", "fmri"]:
        results[mod] = run_baselines(mod)
        
    print("\n\n" + "="*50)
    print("FINAL LINEAR BASELINES BENCHMARK REPORT")
    print("="*50)
    for mod, metrics in results.items():
        print(f"\n{mod.upper()}:")
        print(f"  Ridge: Top-1 {metrics['ridge_t1']:>5.2f}% | Top-5 {metrics['ridge_t5']:>5.2f}%")
        print(f"  CCA:   Top-1 {metrics['cca_t1']:>5.2f}%   | Top-5 {metrics['cca_t5']:>5.2f}%")
    print("="*50)
