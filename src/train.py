import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# Local imports
from src.models.contrastive_model import BrainAlignModel
from src.models.loss import clip_loss
from src.data.eeg_loader import THINGSEEG2Dataset
from src.data.meg_loader import THINGSMEGDataset
from src.data.fmri_loader import THINGSfMRIDataset
from src.evaluate import evaluate

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_dataloader(config, modality, split, subject=1):
    """
    Returns the appropriate DataLoader for the given modality.
    """
    clip_cache_path = os.path.join(config["data"]["clip_cache_dir"], "ViT-B-32.npz")
    batch_size = config["training"]["batch_size"]
    
    if modality == "eeg":
        dataset = THINGSEEG2Dataset(
            eeg_dir=config["data"]["eeg_dir"],
            clip_cache_path=clip_cache_path,
            split=split,
            subject=subject
        )
    elif modality == "meg":
        dataset = THINGSMEGDataset(
            meg_dir=config["data"]["meg_dir"],
            clip_cache_path=clip_cache_path
        )
    elif modality == "fmri":
        dataset = THINGSfMRIDataset(
            fmri_dir=config["data"]["fmri_dir"],
            clip_cache_path=clip_cache_path
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"))

def train(config_path, modality, subject):
    config = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device} for modality {modality.upper()}, subject {subject:02d}")
    
    # Setup data
    print("Initializing dataloader...")
    train_loader = get_dataloader(config, modality, split="train", subject=subject)
    if modality == "eeg":
        test_loader = get_dataloader(config, modality, split="test", subject=subject)
    else:
        test_loader = None
        
    clip_cache_path = os.path.join(config["data"]["clip_cache_dir"], "ViT-B-32.npz")
    clip_dict = np.load(clip_cache_path)
    
    # Inspect first batch to determine input dimension
    # (Since EEG is CxT, fMRI is V, etc... we configure the model dynamically here)
    first_batch = next(iter(train_loader))
    in_shape = first_batch["x"].shape[1:] 
    
    if len(in_shape) == 2:
        in_channels, seq_len = in_shape
    elif len(in_shape) == 1:
        # e.g., fMRI voxels. Mocking channels=1
        in_channels = 1
        seq_len = in_shape[0]
        
    # Setup model
    model = BrainAlignModel(
        in_channels=in_channels,
        seq_len=seq_len,
        brain_embed_dim=config["model"]["projection_dim"],
        clip_dim=512,
        tau_init=config["model"]["temperature_init"]
    ).to(device)
    
    print("Fine-tuning the entire model end-to-end (including the pretrained CBraMod backbone).")
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=float(config["training"]["learning_rate"])
    )
    
    epochs = config["training"]["epochs"]
    best_val_top1 = 0.0
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            x_brain = batch["x"].to(device)
            
            # If input is 1D (fMRI vectors), add dummy channel for Conv1D
            if x_brain.dim() == 2:
                x_brain = x_brain.unsqueeze(1)
                
            y_clip = batch["y_clip"].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: extract projected brain embedding
            p_brain = model(x_brain)
            
            # Loss computation
            loss = clip_loss(p_brain, y_clip, model.logit_scale.to(device))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        
        if test_loader is not None and ((epoch + 1) % 5 == 0 or epoch == 0):
            top1, top5 = evaluate(model, test_loader, clip_dict, device)
            print(f"--> Val Epoch {epoch+1} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}%")
            
            # Only save the best checkpoint
            if top1 > best_val_top1:
                best_val_top1 = top1
                best_ckpt_path = save_dir / f"{modality}_brainalign_sub{subject:02d}_best.pt"
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"    New best validation accuracy! Saved to {best_ckpt_path}")
            
            # Rebalance model to training mode
            model.train()
        
    print(f"Training completed. Best evaluation top-1 metric achieved: {best_val_top1:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BrainAlign Contrastive Logic")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--modality", type=str, required=True, choices=["eeg", "meg", "fmri"], help="Data modality to train on")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID to train on (e.g., 1 for sub-01)")
    args = parser.parse_args()
    
    train(args.config, args.modality, args.subject)
