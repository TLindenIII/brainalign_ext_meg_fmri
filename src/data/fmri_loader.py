import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml
import h5py

from src.data.csv_utils import read_text_table


# Number of highest-variance voxels to retain.
# Top 15K captures ~11% of total variance while removing ~93% of noise voxels.
# This is standard practice in neuroimaging (MindEye, BrainAlign, etc.)
N_TOP_VOXELS = 15000


class THINGSfMRIDataset(Dataset):
    """
    Dataset loader for THINGS-fMRI (ds004192).
    
    Loads single-trial ICA-beta voxel responses from the HDF5 file, selects
    the top-K highest-variance voxels, and preloads everything into RAM for
    fast training.
    
    Using variance-based voxel selection is standard in neuroimaging research:
    most voxels are in non-visual regions and contain only noise. Selecting
    the most variable voxels concentrates the visual signal.
    """
    
    def __init__(
        self,
        fmri_dir,
        clip_cache_path,
        split="train",
        subject=1,
        transform=None,
        shared_only=False,
        quiet=False,
    ):
        self.fmri_dir = Path(fmri_dir)
        self.split = split
        self.transform = transform
        self.subject = subject
        self.shared_only = shared_only
        self.quiet = quiet
        self._log = print if not self.quiet else (lambda *args, **kwargs: None)
        sub_str = f"sub-{subject:02d}"
        
        self._log(f"Loading CLIP cache from {clip_cache_path}")
        self.clip_cache = np.load(clip_cache_path)
        
        self.shared_images = None
        shared_path = Path("data/shared_images.txt")
        if self.shared_only and shared_path.exists():
            with open(shared_path, "r") as f:
                self.shared_images = set([line.strip() for line in f.readlines()])
                self._log(f"Loaded {len(self.shared_images)} shared images for cross-modal filtering")
                
        # ---- Load the stimulus metadata TSV ----
        beta_dir = self.fmri_dir / "derivatives" / "ICA-betas" / sub_str / "voxel-metadata"
        metadata_tsv = beta_dir / f"{sub_str}_task-things_stimulus-metadata.tsv"
        
        if not metadata_tsv.exists():
            raise FileNotFoundError(f"Stimulus metadata not found: {metadata_tsv}")
            
        df = read_text_table(metadata_tsv, expected_columns={"stimulus"})
        self._log(f"Loaded {len(df)} trials from {metadata_tsv.name}")
        
        # ---- Load HDF5, select top-K voxels, preload into RAM ----
        h5_path = beta_dir / f"{sub_str}_task-things_voxel-wise-responses.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Voxel data not found: {h5_path}")
        
        self._log(f"Loading voxel data from {h5_path.name}...")
        with h5py.File(str(h5_path), 'r') as f:
            full_data = f['ResponseData']['block0_values']  # (n_voxels, n_trials)
            n_voxels_full, n_trials_h5 = full_data.shape
            self._log(f"Full HDF5: {n_voxels_full} voxels × {n_trials_h5} trials")
            
            # Select top-K voxels by variance (computed on a subsample for speed)
            self._log(f"Computing per-voxel variance for top-{N_TOP_VOXELS} selection...")
            n_sample = min(3000, n_trials_h5)
            sample_idx = np.linspace(0, n_trials_h5 - 1, n_sample, dtype=int)
            sample_data = full_data[:, sample_idx]
            voxel_var = np.var(sample_data, axis=1)
            
            top_k_indices = np.argsort(voxel_var)[-N_TOP_VOXELS:]
            top_k_indices.sort()  # Keep spatial order for consistency
            
            var_retained = voxel_var[top_k_indices].sum() / voxel_var.sum() * 100
            self._log(f"Selected {N_TOP_VOXELS} voxels ({var_retained:.1f}% of total variance)")
            
            # Preload selected voxels into RAM — (n_trials, n_selected_voxels)
            self._log("Preloading selected voxels into RAM...")
            self.voxel_data = full_data[top_k_indices, :].T.astype(np.float32)  # (n_trials, N_TOP_VOXELS)
        
        self.n_voxels = N_TOP_VOXELS
        self._log(f"Preloaded: {self.voxel_data.shape} ({self.voxel_data.nbytes / 1e6:.0f} MB in RAM)")
        
        # ---- Compute global normalization stats (per-voxel mean/std across all trials) ----
        self.voxel_mean = self.voxel_data.mean(axis=0)  # (N_TOP_VOXELS,)
        self.voxel_std = self.voxel_data.std(axis=0)    # (N_TOP_VOXELS,)
        self.voxel_std[self.voxel_std < 1e-6] = 1.0
            
        # ---- Build trial list ----
        self.trials = []
        for row_idx, row in df.iterrows():
            if row_idx >= n_trials_h5:
                break
                
            stim = str(row.get('stimulus', ''))
            if stim == 'nan' or not stim:
                continue
                
            image_id = Path(stim).stem
            
            if self.shared_images and image_id not in self.shared_images:
                continue
                
            self.trials.append({
                "trial_idx": row_idx,
                "subject": sub_str,
                "image_id": image_id,
            })
                
        # ---- 80/10/10 split on unique image IDs (seeded for cross-modal consistency) ----
        unique_images = sorted(list(set(t["image_id"] for t in self.trials)))
        if not unique_images:
            raise ValueError(
                "No fMRI image IDs remain after preprocessing/filtering. "
                "Check shared-image settings and stimulus metadata."
            )

        np.random.seed(42)
        shuffled = unique_images.copy()
        np.random.shuffle(shuffled)
        
        train_end = int(0.8 * len(shuffled))
        val_end = int(0.9 * len(shuffled))
        
        train_images = set(shuffled[:train_end])
        val_images = set(shuffled[train_end:val_end])
        test_images = set(shuffled[val_end:])
        
        if self.split == "train":
            self.trials = [t for t in self.trials if t["image_id"] in train_images]
        elif self.split == "val":
            self.trials = [t for t in self.trials if t["image_id"] in val_images]
        elif self.split == "test":
            self.trials = [t for t in self.trials if t["image_id"] in test_images]
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")
            
        self._log(
            f"fMRI {sub_str} | {self.split}: {len(self.trials)} trials | "
            f"{self.n_voxels} selected voxels"
        )
        
    def __len__(self):
        return len(self.trials)
        
    def __getitem__(self, idx):
        trial = self.trials[idx]
        image_id = trial["image_id"]
        trial_idx = trial["trial_idx"]
        
        # Fast RAM access — already preloaded and in the right shape
        raw = self.voxel_data[trial_idx]  # (N_TOP_VOXELS,)
        
        # Global z-score normalization (preserves inter-image amplitude differences)
        x = torch.tensor((raw - self.voxel_mean) / self.voxel_std, dtype=torch.float32)
        
        if self.transform:
            x = self.transform(x)
            
        if image_id not in self.clip_cache.files:
            raise KeyError(f"fMRI image_id '{image_id}' not found in CLIP cache")

        y_clip = torch.tensor(self.clip_cache[image_id], dtype=torch.float32)
            
        return {
            "x": x,
            "image_id": image_id,
            "y_clip": y_clip,
            "meta": trial
        }
