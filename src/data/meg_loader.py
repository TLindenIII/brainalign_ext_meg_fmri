import os
from pathlib import Path
import numpy as np
import mne
import torch
from torch.utils.data import Dataset

from src.data.image_manifest import (
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
        quiet=False,
    ):
        self.meg_dir = Path(meg_dir)
        self.split = split
        self.transform = transform
        self.subject = subject
        self.shared_only = shared_only
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
        if merged_file.exists():
            subject_files = [merged_file]
        else:
            subject_files = list(prep_dir.glob(f"preprocessed_P{subject}-epo-*.fif"))

        if not subject_files:
            raise FileNotFoundError(f"No .fif files found for subject {subject} in {prep_dir}")
        subject_files.sort()  # ensure deterministic ordering
            
        self._log(f"Loading {len(subject_files)} preprocessed epoch files for Subject {subject}...")
        
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
        
        # Optional: Normalize channels to Z-score across time standard practice
        # self.meg_data = (self.meg_data - self.meg_data.mean(axis=2, keepdims=True)) / (self.meg_data.std(axis=2, keepdims=True) + 1e-6)
        
        # Enforce intersection filtering
        if self.shared_images:
            self.trials = [t for t in self.trials if t["image_id"] in self.shared_images]
            self._log(f"Retained {len(self.trials)} MEG trials after shared-image filtering")
            
        # Split logic: Seeded 3-way split by fundamental Image ID to avoid data leak.
        unique_images = sorted(list(set(t["image_id"] for t in self.trials)))
        if not unique_images:
            raise ValueError(
                "No MEG image IDs remain after preprocessing/filtering. "
                "Check event mapping and shared-image settings."
            )

        np.random.seed(42)  # Critical for split alignment across MEG/fMRI
        shuffled = unique_images.copy()
        np.random.shuffle(shuffled)
        
        train_end = int(0.8 * len(shuffled))
        val_end = int(0.9 * len(shuffled))
        
        train_images = set(shuffled[:train_end])
        val_images = set(shuffled[train_end:val_end])
        test_images = set(shuffled[val_end:])
        
        self.image_splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        }

        if self.split == "train":
            self.trials = [t for t in self.trials if t["image_id"] in train_images]
        elif self.split == "val":
            self.trials = [t for t in self.trials if t["image_id"] in val_images]
        elif self.split == "test":
            self.trials = [t for t in self.trials if t["image_id"] in test_images]
        elif self.split == "all":
            pass
        else:
            raise ValueError("Split must be 'train', 'val', 'test', or 'all'")
            
        self._log(
            f"MEG Sub-{subject:02d} | {self.split}: {len(self.trials)} trials | "
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
