import os
from pathlib import Path
import numpy as np
import mne
import torch
from torch.utils.data import Dataset

from src.data.image_manifest import (
    ensure_eeg_style_meg_split_lists,
    load_image_split_lists,
    load_named_image_ids,
    load_things_image_map,
    resolve_repo_path,
    resolve_shared_manifest_path,
    resolve_things_image_map_path,
)


class THINGSMEGDataset(Dataset):
    """
    Dataset loader for THINGS-MEG.
    
    Uses mne.read_epochs to load preprocessed .fif files.
    - Each file represents one participant (e.g., preprocessed_P1-epo.fif)
    - Channels: 271 (downsampled/cleaned from original 306)
    - Sample format: [n_epochs, 271 channels, 281 timepoints] (-0.1s to 1.3s @ 200Hz 
      downsampled from 1200Hz).
    - Event codes use THINGS image numbers and must be resolved through a
      THINGS image map before they can be matched to CLIP targets.
    """
    
    def __init__(
        self,
        meg_dir,
        clip_cache_path,
        split="train",
        subject=1,
        transform=None,
        shared_only=False,
        shared_manifest_path=None,
        things_image_map_path=None,
        split_mode="fixed_image_holdout",
        split_manifest_dir=None,
        quiet=False,
    ):
        self.meg_dir = Path(meg_dir)
        self.split = split
        self.split_mode = split_mode
        self.effective_split_mode = split_mode
        self.transform = transform
        self.subject = subject
        self.shared_only = shared_only
        self.split_manifest_dir = resolve_repo_path(split_manifest_dir) if split_manifest_dir else None
        self.quiet = quiet
        self._log = print if not self.quiet else (lambda *args, **kwargs: None)
        
        # Load CLIP cache
        self._log(f"Loading CLIP cache from {clip_cache_path}")
        self.clip_cache = np.load(clip_cache_path)
        
        # Load shared cross-modal image intersection
        self.shared_images = None
        if self.shared_only:
            shared_path = resolve_shared_manifest_path(True, shared_manifest_path)
            self.shared_images = load_named_image_ids(shared_path)
            self._log(f"Loaded {len(self.shared_images)} shared images for cross-modal filtering")

        image_map_path = resolve_things_image_map_path(explicit_path=things_image_map_path)
        if image_map_path is None or not image_map_path.exists():
            raise FileNotFoundError(
                "MEG training requires a full THINGS image map (image number -> image_id). "
                "Expected data/things_image_map.tsv (or another supported map path) by default. "
                "Generate manifests with scripts/build_shared_images.py after providing that map."
            )
        things_image_map = load_things_image_map(image_map_path)
        max_valid_event_id = max(things_image_map) if things_image_map else None
                
        # Discover all preprocessed epoch files for this subject.
        # Prefer the merged subject file when present. Some folders also contain split
        # shards; globbing both the merged file and shards risks double-counting epochs.
        prep_dir = self.meg_dir / "derivatives" / "preprocessed"
        if not prep_dir.exists():
            raise FileNotFoundError(f"MEG preprocessed directory not found: {prep_dir}")

        merged_file = prep_dir / f"preprocessed_P{subject}-epo.fif"
        using_merged_root = merged_file.exists()
        if using_merged_root:
            subject_files = [merged_file]
        else:
            subject_files = list(prep_dir.glob(f"preprocessed_P{subject}-epo-*.fif"))

        if not subject_files:
            raise FileNotFoundError(f"No .fif files found for subject {subject} in {prep_dir}")
        subject_files.sort()  # ensure deterministic ordering
            
        if using_merged_root:
            self._log(
                f"Using merged/root MEG epochs file for Subject {subject}: {merged_file.name} "
                f"(split shards are not loaded separately)."
            )
        else:
            self._log(f"Loading {len(subject_files)} split MEG epoch shard files for Subject {subject}...")
        
        # We will preload everything into RAM to speed up training drastically.
        # A single subject's MEG data is typically ~6GB total.
        all_epochs_data = [] # List of numpy arrays
        self.trials = []
        global_idx = 0
        skipped_unmapped_events = 0
        
        for epo_file in subject_files:
            self._log(f"Reading {epo_file.name}...")
            epochs = mne.read_epochs(str(epo_file), preload=True, verbose='error')
            
            # Modality Conversion Paper Alignment:
            # Crop to -0.5s to 1.0s and resample to 120Hz. We use max/min to safely bound.
            tmin = max(epochs.times[0], -0.5)
            tmax = min(epochs.times[-1], 1.0)
            epochs.crop(tmin=tmin, tmax=tmax)
            epochs.resample(120.0)
            
            # epochs.get_data() returns numpy array (epochs, channels, times)
            data = epochs.get_data(copy=False).astype(np.float32)
            all_epochs_data.append(data)
            
            # Parse events array [frame, zero, ID]
            events = epochs.events
            
            for i, event_row in enumerate(events):
                data_row_idx = global_idx
                global_idx += 1
                event_id = int(event_row[2])
                
                # Check mapping
                mapped = things_image_map.get(event_id)
                if mapped is None:
                    skipped_unmapped_events += 1
                    continue
                image_id = mapped["image_id"]
                    
                self.trials.append({
                    "global_idx": data_row_idx, # Map back to the concatenated numpy array
                    "image_id": image_id,
                    "event_id": event_id
                })
                
        # Concatenate MEG continuous arrays into single RAM block
        self._log("Concatenating MEG epochs into RAM...")
        self.meg_data = np.concatenate(all_epochs_data, axis=0)
        self.n_channels, self.seq_len = self.meg_data.shape[1], self.meg_data.shape[2]
        self._log(f"Preloaded: {self.meg_data.shape} ({self.meg_data.nbytes / 1e6:.0f} MB RAM)")
        if max_valid_event_id is not None and skipped_unmapped_events > 0:
            self._log(
                f"Skipped {skipped_unmapped_events} unmapped MEG events "
                f"(not present in the THINGS image map up to {max_valid_event_id}, e.g. button-press markers)."
            )

        # Match the reference preprocessing more closely by standardizing each
        # channel within each epoch after cropping/resampling.
        self._log("Centering and standardizing MEG epochs channel-wise...")
        epoch_mean = self.meg_data.mean(axis=2, keepdims=True, dtype=np.float32)
        epoch_std = self.meg_data.std(axis=2, keepdims=True, dtype=np.float32)
        np.maximum(epoch_std, 1e-6, out=epoch_std)
        self.meg_data -= epoch_mean
        self.meg_data /= epoch_std
        np.nan_to_num(self.meg_data, copy=False)
        
        # Enforce intersection filtering
        if self.shared_images:
            self.trials = [t for t in self.trials if t["image_id"] in self.shared_images]
            self._log(f"Retained {len(self.trials)} MEG trials after shared-image filtering")
            
        unique_images = sorted(list(set(t["image_id"] for t in self.trials)))
        if not unique_images:
            raise ValueError(
                "No MEG image IDs remain after preprocessing/filtering. "
                "Check event mapping and shared-image settings."
            )
        self.image_splits = self._build_image_splits(unique_images)

        if self.split == "train":
            self.trials = [t for t in self.trials if t["image_id"] in self.image_splits["train"]]
        elif self.split == "val":
            self.trials = [t for t in self.trials if t["image_id"] in self.image_splits["val"]]
        elif self.split == "test":
            self.trials = [t for t in self.trials if t["image_id"] in self.image_splits["test"]]
        elif self.split == "all":
            pass
        else:
            raise ValueError("Split must be 'train', 'val', 'test', or 'all'")
            
        self._log(
            f"MEG Sub-{subject:02d} | {self.split} ({self.effective_split_mode}): {len(self.trials)} trials | "
            f"Channels: {self.n_channels}, SeqLen: {self.seq_len}"
        )
        
    def __len__(self):
        return len(self.trials)
        
    def __getitem__(self, idx):
        trial = self.trials[idx]
        image_id = trial["image_id"]
        global_idx = trial["global_idx"]
        
        # Fast read from memory
        x = torch.tensor(self.meg_data[global_idx], dtype=torch.float32)
        
        if self.transform:
            x = self.transform(x)
            
        if image_id not in self.clip_cache.files:
            raise KeyError(f"MEG image_id '{image_id}' not found in CLIP cache")

        y_clip = torch.tensor(self.clip_cache[image_id], dtype=torch.float32)
            
        return {
            "x": x,
            "image_id": image_id,
            "y_clip": y_clip,
            "meta": trial
        }

    def _build_image_splits(self, unique_images):
        if self.split_mode == "fixed_image_holdout":
            return self._fixed_image_holdout_splits(unique_images)
        if self.split_mode == "random_strict":
            return self._random_strict_splits(unique_images)
        raise ValueError(
            f"Unsupported MEG split mode '{self.split_mode}'. "
            "Expected 'fixed_image_holdout' or 'random_strict'."
        )

    def _fixed_image_holdout_splits(self, unique_images):
        split_dir = self.split_manifest_dir or Path("data/manifests/splits/meg/fixed_image_holdout")
        ensure_eeg_style_meg_split_lists(
            split_dir,
            unique_images,
            seed=42,
            test_concept_count=200,
            val_ratio=0.1,
            overwrite=False,
        )
        saved_splits = load_image_split_lists(split_dir)
        unique_image_set = set(unique_images)
        image_splits = {
            split_name: set(image_ids) & unique_image_set
            for split_name, image_ids in saved_splits.items()
        }

        assigned = set().union(*image_splits.values())
        excluded_path = split_dir / "excluded.txt"
        excluded_images = load_named_image_ids(excluded_path) if excluded_path.exists() else set()
        missing = unique_image_set - assigned - excluded_images
        if missing:
            raise ValueError(
                f"MEG fixed-image split manifests under {split_dir} do not cover "
                f"{len(missing)} current images. Rebuild the manifests."
            )

        if excluded_images:
            self._log(f"Excluded {len(excluded_images & unique_image_set)} MEG images from held-out test concepts")
        self._log(f"Using MEG fixed-image split manifests from {split_dir}")
        return image_splits

    def _random_strict_splits(self, unique_images):
        rng = np.random.RandomState(42)
        shuffled = unique_images.copy()
        rng.shuffle(shuffled)

        train_end = int(0.8 * len(shuffled))
        val_end = int(0.9 * len(shuffled))
        self.effective_split_mode = "random_strict"
        return {
            "train": set(shuffled[:train_end]),
            "val": set(shuffled[train_end:val_end]),
            "test": set(shuffled[val_end:]),
        }
