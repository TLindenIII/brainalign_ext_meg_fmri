import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from src.models.contrastive_model import BrainAlignModel
from src.data.eeg_loader import THINGSEEG2Dataset
from src.data.meg_loader import THINGSMEGDataset
from src.data.fmri_loader import THINGSfMRIDataset

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate(model, test_loader, clip_dict, device):
    model.eval()
    
    # 1. Prepare candidates
    unique_test_ids = list(set([Path(f).stem for f in test_loader.dataset.files]))
    
    test_candidates = []
    for cid in unique_test_ids:
        test_candidates.append(clip_dict[cid])
        
    test_candidates = np.array(test_candidates)
    
    # Accumulate projected brain embeddings per image ID
    target_p_brains = {tid: [] for tid in unique_test_ids}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            x_brain = batch["x"].to(device)
            if x_brain.dim() == 2:
                x_brain = x_brain.unsqueeze(1)
                
            id_targets = batch['image_id']
            p_brain = model(x_brain)
            
            # Predict
            Y_pred = p_brain.cpu().numpy()
            
            for i, true_id in enumerate(id_targets):
                target_p_brains[true_id].append(Y_pred[i])
                
    # Average the projected representations per image condition as specified in the paper
    top1, top5, total = 0, 0, 0
    
    for true_id, p_brain_trials in target_p_brains.items():
        if len(p_brain_trials) == 0:
            continue
            
        # Average the representation for this true_id
        avg_p_brain = np.mean(p_brain_trials, axis=0, keepdims=True)
        
        # Distance against all 200 candidates
        sims = cosine_similarity(avg_p_brain, test_candidates)
        sorted_indices = np.argsort(sims, axis=1)[:, ::-1]
        
        best_5 = [unique_test_ids[idx] for idx in sorted_indices[0][:5]]
        if true_id == best_5[0]: top1 += 1
        if true_id in best_5: top5 += 1
        total += 1
                
    return (top1/total)*100 if total > 0 else 0, (top5/total)*100 if total > 0 else 0


def main(modality, checkpoint_path):
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device} to evaluate {modality}")
    
    clip_cache_path = os.path.join(config["data"]["clip_cache_dir"], "ViT-B-32.npz")
    clip_dict = np.load(clip_cache_path)
    
    # Setup Data
    if modality == "eeg":
        dataset = THINGSEEG2Dataset(eeg_dir=config["data"]["eeg_dir"], clip_cache_path=clip_cache_path, split="test")
    elif modality == "meg":
        dataset = THINGSMEGDataset(meg_dir=config["data"]["meg_dir"], clip_cache_path=clip_cache_path)
    elif modality == "fmri":
        dataset = THINGSfMRIDataset(fmri_dir=config["data"]["fmri_dir"], clip_cache_path=clip_cache_path)
        
    test_loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    
    first_batch = next(iter(test_loader))
    in_shape = first_batch["x"].shape[1:] 
    
    if len(in_shape) == 2:
        in_channels, seq_len = in_shape
    elif len(in_shape) == 1:
        in_channels = 1
        seq_len = in_shape[0]
        
    model = BrainAlignModel(
        in_channels=in_channels,
        seq_len=seq_len,
        brain_embed_dim=512, # Restore to match the 512x512 projection checkpoints from the frozen runs
        clip_dim=512,
        tau_init=config["model"]["temperature_init"]
    ).to(device)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    
    print("Beginning retrieval evaluation...")
    top1, top5 = evaluate(model, test_loader, clip_dict, device)
    
    print(f"\n--- Evaluation Results ({modality.upper()}) ---")
    out_lines = [
        f"--- Evaluation Results ({modality.upper()}) ---",
        f"Checkpoint: {checkpoint_path}",
        f"Top-1 Retrieval: {top1:.2f}%",
        f"Top-5 Retrieval: {top5:.2f}%"
    ]
    for line in out_lines:
        print(line)
        
    results_dir = Path("results") / modality
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "evaluation_results.txt", "w") as f:
        f.write("\n".join(out_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()
    main(args.modality, args.ckpt)
