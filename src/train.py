import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
# Local imports
from src.models.contrastive_model import BrainAlignModel
from src.models.fmri_model import fMRIAlignModel
from src.models.loss import clip_loss
from src.data.eeg_loader import THINGSEEG2Dataset
from src.data.meg_loader import THINGSMEGDataset
from src.data.fmri_loader import THINGSfMRIDataset
from src.evaluate import evaluate


def checkpoint_stem_for(modality, subject, shared_only=False):
    stem = f"{modality}_brainalign_sub{subject:02d}"
    if modality == "meg":
        stem += "_attnpool"
    if shared_only:
        stem += "_shared"
    return stem

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_dataloader(config, modality, split, subject=1, shared_only=False):
    """
    Returns the appropriate DataLoader for the given modality.
    """
    clip_cache_path = os.path.join(config["data"]["clip_cache_dir"], "ViT-B-32.npz")
    shared_manifest_path = config["data"].get("shared_manifest_path")
    things_image_map_path = config["data"].get("things_image_map_path")
    
    if modality == "eeg":
        dataset = THINGSEEG2Dataset(
            eeg_dir=config["data"]["eeg_dir"],
            clip_cache_path=clip_cache_path,
            split=split,
            subject=subject,
            shared_only=shared_only,
            shared_manifest_path=shared_manifest_path,
        )
    elif modality == "meg":
        dataset = THINGSMEGDataset(
            meg_dir=config["data"]["meg_dir"],
            clip_cache_path=clip_cache_path,
            split=split,
            subject=subject,
            shared_only=shared_only,
            shared_manifest_path=shared_manifest_path,
            things_image_map_path=things_image_map_path,
        )
    elif modality == "fmri":
        dataset = THINGSfMRIDataset(
            fmri_dir=config["data"]["fmri_dir"],
            clip_cache_path=clip_cache_path,
            split=split,
            subject=subject,
            shared_only=shared_only,
            shared_manifest_path=shared_manifest_path,
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")
        
    batch_size = config["training"]["batch_size"][modality]
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"))


def get_meg_train_val_dataloaders(config, subject=1, shared_only=False):
    clip_cache_path = os.path.join(config["data"]["clip_cache_dir"], "ViT-B-32.npz")
    full_dataset = THINGSMEGDataset(
        meg_dir=config["data"]["meg_dir"],
        clip_cache_path=clip_cache_path,
        split="all",
        subject=subject,
        shared_only=shared_only,
        shared_manifest_path=config["data"].get("shared_manifest_path"),
        things_image_map_path=config["data"].get("things_image_map_path"),
    )

    train_images = full_dataset.image_splits["train"]
    val_images = full_dataset.image_splits["val"]
    train_indices = [idx for idx, trial in enumerate(full_dataset.trials) if trial["image_id"] in train_images]
    val_indices = [idx for idx, trial in enumerate(full_dataset.trials) if trial["image_id"] in val_images]

    batch_size = config["training"]["batch_size"]["meg"]
    train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=batch_size, shuffle=False)
    print(
        f"Reused one MEG preload for train/val: "
        f"{len(train_indices)} train trials, {len(val_indices)} val trials"
    )
    return train_loader, val_loader

def train(
    config_path,
    modality,
    subject,
    epochs_override=None,
    resume=False,
    resume_best=False,
    shared_only=False,
    shared_manifest_path=None,
):
    config = load_config(config_path)
    if shared_manifest_path:
        config.setdefault("data", {})["shared_manifest_path"] = shared_manifest_path
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    scope_label = "shared" if shared_only else "full"
    print(f"Using device: {device} for modality {modality.upper()}, subject {subject:02d} ({scope_label})")
    
    # Setup data
    print("Initializing dataloader...")
    if modality == "meg":
        train_loader, val_loader = get_meg_train_val_dataloaders(
            config,
            subject=subject,
            shared_only=shared_only,
        )
    else:
        train_loader = get_dataloader(config, modality, split="train", subject=subject, shared_only=shared_only)
        val_loader = get_dataloader(config, modality, split="val", subject=subject, shared_only=shared_only)
        
    clip_cache_path = os.path.join(config["data"]["clip_cache_dir"], "ViT-B-32.npz")
    clip_dict = np.load(clip_cache_path)
    
    # Inspect first batch to determine input dimension
    # (Since EEG is CxT, fMRI is V, etc... we configure the model dynamically here)
    first_batch = next(iter(train_loader))
    in_shape = first_batch["x"].shape[1:] 
    
    if len(in_shape) == 2:
        in_channels, seq_len = in_shape
    elif len(in_shape) == 1:
        # e.g., fMRI voxels — a flat 1D vector
        in_channels = 1
        seq_len = in_shape[0]
        
    # Setup model — use dedicated fMRI model for voxel data, CBraMod for time series
    if modality == "fmri":
        n_voxels = in_shape[0]
        model = fMRIAlignModel(
            n_voxels=n_voxels,
            clip_dim=512,
            tau_init=config["model"]["temperature_init"]
        ).to(device)
        print(f"Using fMRIAlignModel: {n_voxels} voxels → 512-dim CLIP space")
    else:
        model = BrainAlignModel(
            in_channels=in_channels,
            seq_len=seq_len,
            brain_embed_dim=config["model"]["projection_dim"],
            clip_dim=512,
            tau_init=config["model"]["temperature_init"],
            modality=modality,
        ).to(device)
    
    print("Fine-tuning the entire model end-to-end (including the pretrained CBraMod backbone).")
    lr = float(config["training"]["learning_rate"])
    
    # BrainAlign and Modality Conversion both specify AdamW with 1e-3 weight decay for ALL models
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=1e-3
    )
    print(f"Using AdamW optimizer with lr={lr:.1e}, weight_decay=1e-3")
    
    epochs = config["training"]["epochs"][modality] if epochs_override is None else epochs_override
    best_val_metric = 0.0
    save_dir = Path("checkpoints") / modality
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = checkpoint_stem_for(modality, subject, shared_only=shared_only)
    best_ckpt_path = save_dir / f"{checkpoint_stem}_best.pt"
    latest_ckpt_path = save_dir / f"{checkpoint_stem}_latest.pt"
    
    start_epoch = 0
    if resume or resume_best:
        target_ckpt = None
        
        # Priority: resume_best flag -> latest_ckpt -> best_ckpt
        if resume_best and best_ckpt_path.exists():
            target_ckpt = best_ckpt_path
            print(f"Resume-best flag detected. Using 'best' checkpoint: {target_ckpt}")
        elif latest_ckpt_path.exists():
            target_ckpt = latest_ckpt_path
            print(f"Resume flag detected. Found 'latest' checkpoint: {target_ckpt}")
        elif best_ckpt_path.exists():
            target_ckpt = best_ckpt_path
            print(f"Resume flag detected. Found 'best' checkpoint: {target_ckpt}")
            
        if target_ckpt:
            checkpoint = torch.load(target_ckpt, map_location=device)
            
            # Check if it's the new comprehensive dict format or legacy raw weights
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                best_val_metric = checkpoint.get("best_val_metric", checkpoint.get("best_val_top1", 0.0))
                print(f"Restored comprehensive checkpoint (Epoch {start_epoch-1}, Best Val: {best_val_metric:.2f}%)")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded legacy weights-only checkpoint. Starting from Epoch 1.")
        else:
            print("Resume flag detected but no checkpoints found. Starting fresh.")
            
    print(f"Starting training loop from epoch {start_epoch+1} to {epochs}...")
    for epoch in range(start_epoch, epochs):
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
            
            # Safety guard: skip batch if loss is NaN (e.g. from corrupted/infinite inputs)
            if not torch.isfinite(loss):
                pbar.set_postfix({"Loss": "NaN - skipped"})
                optimizer.zero_grad()
                continue
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}")
        
        if val_loader is not None:
            metrics = evaluate(model, val_loader, clip_dict, device)
            top1 = metrics['top1']
            top5 = metrics['top5']
            two_way = metrics['two_way']
            print(f"--> Val Epoch {epoch+1} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | 2-way: {two_way:.2f}%")
            
            # Use 2-way accuracy for fMRI, Top-1 for others as the primary metric
            current_metric = two_way if modality == "fmri" else top1
            
            # Only save the best checkpoint
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_metric': best_val_metric,
                }, best_ckpt_path)
                print(f"    New best validation metric ({current_metric:.2f}%)! Saved comprehensive statemap to {best_ckpt_path}")
            
            # Rebalance model to training mode
            model.train()
            
        # Always save the latest state dictionary at the very end of the epoch loop
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_metric': best_val_metric,
        }, latest_ckpt_path)
        
    print(f"Training completed. Best evaluation metric achieved: {best_val_metric:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BrainAlign Contrastive Logic")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--modality", type=str, required=True, choices=["eeg", "meg", "fmri"], help="Data modality to train on")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID to train on (e.g., 1 for sub-01)")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of epochs (default: config.yaml value)")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint if it exists")
    parser.add_argument("--resume-best", action="store_true", help="Resume from the best checkpoint instead of the latest")
    parser.add_argument(
        "--shared-only",
        action="store_true",
        help="Restrict MEG/fMRI training to the shared image intersection used for conversion",
    )
    parser.add_argument(
        "--shared-manifest",
        type=str,
        default=None,
        help="Optional manifest of image_ids to use when --shared-only is enabled",
    )
    args = parser.parse_args()
    
    train(
        args.config,
        args.modality,
        args.subject,
        args.epochs,
        args.resume,
        args.resume_best,
        args.shared_only,
        args.shared_manifest,
    )
