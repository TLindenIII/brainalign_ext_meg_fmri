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
from src.models.meg_model import MEGAlignModel
from src.models.loss import clip_loss
from src.data.eeg_loader import THINGSEEG2Dataset
from src.data.image_manifest import conversion_split_dir_from_config
from src.data.meg_loader import THINGSMEGDataset
from src.data.fmri_loader import THINGSfMRIDataset
from src.checkpoints import checkpoint_paths_for, conversion_manifest_slug
from src.evaluate import evaluate

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def selection_metric_name(config, modality):
    configured = config.get("training", {}).get("selection_metric", {}).get(modality)
    if configured:
        return configured
    return "top1" if modality == "eeg" else "mrr"


def selection_metric_value(metrics, metric_name):
    if metric_name not in metrics:
        raise KeyError(f"Metric '{metric_name}' not available in evaluation output")
    return metrics[metric_name]

def get_dataloader(config, modality, split, subject=1, shared_only=False):
    """
    Returns the appropriate DataLoader for the given modality.
    """
    clip_cache_path = os.path.join(config["data"]["clip_cache_dir"], "ViT-B-32.npz")
    shared_manifest_path = config["data"].get("shared_manifest_path")
    things_image_map_path = config["data"].get("things_image_map_path")
    manifests_dir = config["data"].get("manifests_dir", "data/manifests")
    meg_split_mode = config["data"].get("meg_split_mode", "fixed_image_holdout")
    fmri_split_mode = config["data"].get("fmri_split_mode", "official_repeats")
    conversion_config = config.get("conversion", {})
    shared_split_dir = None
    if shared_only and shared_manifest_path:
        shared_split_dir = conversion_split_dir_from_config(
            config,
            shared_manifest_path=shared_manifest_path,
        )
    
    if modality == "eeg":
        dataset = THINGSEEG2Dataset(
            eeg_dir=config["data"]["eeg_dir"],
            clip_cache_path=clip_cache_path,
            split=split,
            subject=subject,
            shared_only=shared_only,
            shared_manifest_path=shared_manifest_path,
            shared_split_dir=shared_split_dir,
            shared_split_seed=conversion_config.get("split_seed", 42),
            shared_val_concept_count=conversion_config.get("val_concepts", 100),
            shared_test_concept_count=conversion_config.get("test_concepts", 200),
        )
    elif modality == "meg":
        dataset = THINGSMEGDataset(
            meg_dir=config["data"]["meg_dir"],
            clip_cache_path=clip_cache_path,
            split=split,
            subject=subject,
            shared_only=shared_only,
            shared_manifest_path=shared_manifest_path,
            shared_split_dir=shared_split_dir,
            shared_split_seed=conversion_config.get("split_seed", 42),
            shared_val_concept_count=conversion_config.get("val_concepts", 100),
            shared_test_concept_count=conversion_config.get("test_concepts", 200),
            things_image_map_path=things_image_map_path,
            split_mode=meg_split_mode,
            split_manifest_dir=os.path.join(manifests_dir, "splits", "meg", meg_split_mode),
        )
    elif modality == "fmri":
        dataset = THINGSfMRIDataset(
            fmri_dir=config["data"]["fmri_dir"],
            clip_cache_path=clip_cache_path,
            split=split,
            subject=subject,
            shared_only=shared_only,
            shared_manifest_path=shared_manifest_path,
            shared_split_dir=shared_split_dir,
            shared_split_seed=conversion_config.get("split_seed", 42),
            shared_val_concept_count=conversion_config.get("val_concepts", 100),
            shared_test_concept_count=conversion_config.get("test_concepts", 200),
            split_mode=fmri_split_mode,
        )
    else:
        raise ValueError(f"Unknown modality: {modality}")
        
    batch_size = config["training"]["batch_size"][modality]
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"))


def get_meg_train_val_dataloaders(config, subject=1, shared_only=False):
    clip_cache_path = os.path.join(config["data"]["clip_cache_dir"], "ViT-B-32.npz")
    shared_manifest_path = config["data"].get("shared_manifest_path")
    conversion_config = config.get("conversion", {})
    shared_split_dir = None
    if shared_only and shared_manifest_path:
        shared_split_dir = conversion_split_dir_from_config(
            config,
            shared_manifest_path=shared_manifest_path,
        )
    full_dataset = THINGSMEGDataset(
        meg_dir=config["data"]["meg_dir"],
        clip_cache_path=clip_cache_path,
        split="all",
        subject=subject,
        shared_only=shared_only,
        shared_manifest_path=shared_manifest_path,
        shared_split_dir=shared_split_dir,
        shared_split_seed=conversion_config.get("split_seed", 42),
        shared_val_concept_count=conversion_config.get("val_concepts", 100),
        shared_test_concept_count=conversion_config.get("test_concepts", 200),
        things_image_map_path=config["data"].get("things_image_map_path"),
        split_mode=config["data"].get("meg_split_mode", "fixed_image_holdout"),
        split_manifest_dir=os.path.join(
            config["data"].get("manifests_dir", "data/manifests"),
            "splits",
            "meg",
            config["data"].get("meg_split_mode", "fixed_image_holdout"),
        ),
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


def get_fmri_train_val_dataloaders(config, subject=1, shared_only=False):
    clip_cache_path = os.path.join(config["data"]["clip_cache_dir"], "ViT-B-32.npz")
    shared_manifest_path = config["data"].get("shared_manifest_path")
    conversion_config = config.get("conversion", {})
    shared_split_dir = None
    if shared_only and shared_manifest_path:
        shared_split_dir = conversion_split_dir_from_config(
            config,
            shared_manifest_path=shared_manifest_path,
        )
    full_dataset = THINGSfMRIDataset(
        fmri_dir=config["data"]["fmri_dir"],
        clip_cache_path=clip_cache_path,
        split="all",
        subject=subject,
        shared_only=shared_only,
        shared_manifest_path=shared_manifest_path,
        shared_split_dir=shared_split_dir,
        shared_split_seed=conversion_config.get("split_seed", 42),
        shared_val_concept_count=conversion_config.get("val_concepts", 100),
        shared_test_concept_count=conversion_config.get("test_concepts", 200),
        split_mode=config["data"].get("fmri_split_mode", "official_repeats"),
    )

    train_indices = [
        idx for idx, trial in enumerate(full_dataset.trials) if trial["assigned_split"] == "train"
    ]
    val_indices = [
        idx for idx, trial in enumerate(full_dataset.trials) if trial["assigned_split"] == "val"
    ]

    batch_size = config["training"]["batch_size"]["fmri"]
    train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=batch_size, shuffle=False)
    print(
        f"Reused one fMRI preload for train/val: "
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
    if shared_only and not config.get("data", {}).get("shared_manifest_path"):
        raise ValueError("Shared conversion training requires an explicit --shared-manifest")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    scope_label = (
        f"shared:{conversion_manifest_slug(shared_manifest_path=config['data']['shared_manifest_path'])}"
        if shared_only
        else "full"
    )
    print(f"Using device: {device} for modality {modality.upper()}, subject {subject:02d} ({scope_label})")
    
    # Setup data
    print("Initializing dataloader...")
    if modality == "meg":
        train_loader, val_loader = get_meg_train_val_dataloaders(
            config,
            subject=subject,
            shared_only=shared_only,
        )
    elif modality == "fmri":
        train_loader, val_loader = get_fmri_train_val_dataloaders(
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
    elif modality == "meg":
        model = MEGAlignModel(
            in_channels=in_channels,
            seq_len=seq_len,
            clip_dim=512,
            hidden_dim=config["model"].get("meg_hidden_dim", 256),
            dropout=config["model"].get("meg_dropout", 0.2),
            tau_init=config["model"]["temperature_init"],
        ).to(device)
        print(
            f"Using MEGAlignModel: {in_channels} sensors × {seq_len} timepoints "
            f"→ {config['model'].get('meg_hidden_dim', 256)} hidden channels"
        )
    else:
        model = BrainAlignModel(
            in_channels=in_channels,
            seq_len=seq_len,
            brain_embed_dim=config["model"]["projection_dim"],
            clip_dim=512,
            tau_init=config["model"]["temperature_init"],
            modality=modality,
        ).to(device)
    
    if modality == "eeg":
        print("Fine-tuning the entire model end-to-end (including the pretrained CBraMod backbone).")
    else:
        print("Training the modality-specific alignment model end-to-end.")
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
    checkpoint_paths = checkpoint_paths_for(
        modality,
        subject,
        shared_only=shared_only,
        shared_manifest_path=config["data"].get("shared_manifest_path"),
    )
    save_dir = checkpoint_paths["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = checkpoint_paths["best"]
    latest_ckpt_path = checkpoint_paths["latest"]
    print(f"Checkpoint directory: {save_dir}")
    
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
            mrr = metrics['mrr']
            mean_rank = metrics['mean_rank']
            metric_name = selection_metric_name(config, modality)
            current_metric = selection_metric_value(metrics, metric_name)
            print(
                f"--> Val Epoch {epoch+1} | Top-1: {top1:.2f}% | Top-5: {top5:.2f}% | "
                f"2-way: {two_way:.2f}% | MRR: {mrr:.2f}% | Mean rank: {mean_rank:.2f} | "
                f"Select: {metric_name}={current_metric:.2f}"
            )
            
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
