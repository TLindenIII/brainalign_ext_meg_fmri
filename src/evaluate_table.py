import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from src.models.contrastive_model import BrainAlignModel
from src.data.eeg_loader import THINGSEEG2Dataset

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate_subject(model, test_loader, clip_dict, device):
    model.eval()
    
    unique_test_ids = list(set([Path(f).stem for f in test_loader.dataset.files]))
    unique_test_ids.sort() # Ensure consistent ordering mapping the 200 class indices to the arrays
    
    test_candidates = []
    for cid in unique_test_ids:
        test_candidates.append(clip_dict[cid])
    test_candidates = np.array(test_candidates) # Shape: (200, 512)
    
    target_p_brains = {tid: [] for tid in unique_test_ids}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            x_brain = batch["x"].to(device)
            if x_brain.dim() == 2:
                x_brain = x_brain.unsqueeze(1)
                
            id_targets = batch['image_id']
            p_brain = model(x_brain)
            Y_pred = p_brain.cpu().numpy()
            
            for i, true_id in enumerate(id_targets):
                target_p_brains[true_id].append(Y_pred[i])
                
    # Average representations for each of the 200 condition IDs
    avg_p_brains = []
    
    for tid in unique_test_ids:
        trials = target_p_brains[tid]
        if len(trials) == 0:
            avg_p_brains.append(np.zeros(512)) # fallback
        else:
            avg_p_brains.append(np.mean(trials, axis=0))
            
    avg_p_brains = np.array(avg_p_brains) # Shape: (200, 512)
    
    # EEG-to-Image similarity: queries are EEG, gallery are Images
    sims_e2i = cosine_similarity(avg_p_brains, test_candidates) # shape: (200, 200)
    top1_e2i = 0
    for i in range(200):
        # The correct match is the element on the diagonal where row i matches col i
        sorted_indices = np.argsort(sims_e2i[i])[::-1]
        if sorted_indices[0] == i:
            top1_e2i += 1
    e2i_acc = (top1_e2i / 200.0) * 100.0
    
    # Image-to-EEG similarity: queries are Images, gallery are EEG
    sims_i2e = cosine_similarity(test_candidates, avg_p_brains) # shape: (200, 200)
    top1_i2e = 0
    for i in range(200):
        sorted_indices = np.argsort(sims_i2e[i])[::-1]
        if sorted_indices[0] == i:
            top1_i2e += 1
    i2e_acc = (top1_i2e / 200.0) * 100.0
    
    return e2i_acc, i2e_acc

def main(subject):
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device} to evaluate tables")
    
    clip_cache_path = os.path.join(config["data"]["clip_cache_dir"], "ViT-B-32.npz")
    clip_dict = np.load(clip_cache_path)
    
    results_e2i = []
    results_i2e = []
    
    subjects_to_eval = [subject] if subject is not None else range(1, 11)
    
    # We load a sample dataset just to extract the input dimensions
    dataset1 = THINGSEEG2Dataset(eeg_dir=config["data"]["eeg_dir"], clip_cache_path=clip_cache_path, split="test", subject=1)
    loader1 = DataLoader(dataset1, batch_size=1, shuffle=False)
    batch = next(iter(loader1))
    in_shape = batch["x"].shape[1:]
    in_channels, seq_len = in_shape
    
    print(f"Beginning evaluation across {len(subjects_to_eval)} subjects...")
    for sub in subjects_to_eval:
        checkpoint_path = f"checkpoints/eeg_brainalign_sub{sub:02d}_best.pt"
        
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint {checkpoint_path} not found. Skipping subject {sub:02d} and assigning 0.0 accuracy.")
            results_e2i.append(0.0)
            results_i2e.append(0.0)
            continue
            
        model = BrainAlignModel(
            in_channels=in_channels,
            seq_len=seq_len,
            brain_embed_dim=512,
            clip_dim=512,
            tau_init=config["model"]["temperature_init"]
        ).to(device)
            
        print(f"Loading checkpoint for subject {sub:02d}: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        
        dataset = THINGSEEG2Dataset(
            eeg_dir=config["data"]["eeg_dir"], 
            clip_cache_path=clip_cache_path, 
            split="test",
            subject=sub
        )
        test_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)
            
        e2i_acc, i2e_acc = evaluate_subject(model, test_loader, clip_dict, device)
        results_e2i.append(e2i_acc)
        results_i2e.append(i2e_acc)
        
    e2i_arr = np.array(results_e2i)
    i2e_arr = np.array(results_i2e)
    
    e2i_mean = e2i_arr.mean()
    e2i_std = e2i_arr.std(ddof=1) if len(e2i_arr) > 1 else 0.0
    
    i2e_mean = i2e_arr.mean()
    i2e_std = i2e_arr.std(ddof=1) if len(i2e_arr) > 1 else 0.0
    
    def format_row(method_name, results_arr, mean, std):
        row_str = f"{method_name:<30} "
        for val in results_arr:
            row_str += f"{val:5.1f} "
        row_str += f"{mean:6.1f} {std:5.1f}"
        return row_str
        
    print("\nTable 1: A comparison of different model performances (top-1 accuracies) across evaluated subjects for the EEG-to-Image 200-way zero-shot classification task")
    header = f"{'Method':<30} " + " ".join([f"S{s:<4}" for s in subjects_to_eval]) + f"{'Ave':>7} {'Std':>6}"
    print(header)
    print(format_row("Our CBraMod (finetuned) + CLIP", e2i_arr, e2i_mean, e2i_std))
    
    print("\nTable 2: A comparison of different model performances (top-1 accuracies) across evaluated subjects for the Image-to-EEG 200-way zero-shot classification task")
    print(header)
    print(format_row("Our CBraMod (finetuned) + CLIP", i2e_arr, i2e_mean, i2e_std))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=None, help="Evaluate a single subject (e.g., 1). If not provided, loop 1-10.")
    args = parser.parse_args()
    main(args.subject)
