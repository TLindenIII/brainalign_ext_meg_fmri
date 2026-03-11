import os
from pathlib import Path
import numpy as np
import mne
import torch
from torch.utils.data import Dataset


class THINGSMEGDataset(Dataset):
    """
    Dataset loader for THINGS-MEG.
    
    Uses mne.read_epochs to load preprocessed .fif files.
    - Each file represents one participant (e.g., preprocessed_P1-epo.fif)
    - Channels: 271 (downsampled/cleaned from original 306)
    - Sample format: [n_epochs, 271 channels, 281 timepoints] (-0.1s to 1.3s @ 200Hz 
      downsampled from 1200Hz).
    - Event codes map to the global 1-16540 image stimuli IDs.
    """
    
    def __init__(self, meg_dir, clip_cache_path, split="train", subject=1, transform=None):
        self.meg_dir = Path(meg_dir)
        self.split = split
        self.transform = transform
        self.subject = subject
        
        # Load CLIP cache
        print(f"Loading CLIP cache from {clip_cache_path}")
        self.clip_cache = np.load(clip_cache_path)
        
        # Load shared cross-modal image intersection
        self.shared_images = None
        shared_path = Path("data/shared_images.txt")
        if shared_path.exists():
            with open(shared_path, "r") as f:
                self.shared_images = set([line.strip() for line in f.readlines()])
                print(f"Loaded {len(self.shared_images)} shared images for cross-modal filtering")
                
        # To strictly map MNE event integers to global IDs as strings
        id_to_string_map = {}
        metadata_path = Path("data/things-eeg2/stimuli/image_metadata.npy")
        if metadata_path.exists():
            metadata = np.load(metadata_path, allow_pickle=True).item()
            train_files = [Path(f).stem for f in metadata["train_img_files"]]
            test_files = [Path(f).stem for f in metadata["test_img_files"]]
            all_files = train_files + test_files
            id_to_string_map = {str(i+1): file_stem for i, file_stem in enumerate(all_files)}
                
        # Discover all preprocessed epoch files for this subject.
        # Format Example: preprocessed_P1-epo.fif (Sub 1 may have multiple splits like -1, -2)
        prep_dir = self.meg_dir / "derivatives" / "preprocessed"
        if not prep_dir.exists():
            raise FileNotFoundError(f"MEG preprocessed directory not found: {prep_dir}")
            
        subject_files = list(prep_dir.glob(f"preprocessed_P{subject}-epo*.fif"))
        if not subject_files:
            raise FileNotFoundError(f"No .fif files found for subject {subject} in {prep_dir}")
        subject_files.sort()  # ensure deterministic ordering
            
        print(f"Loading {len(subject_files)} preprocessed epoch files for Subject {subject}...")
        
        # We will preload everything into RAM to speed up training drastically.
        # A single subject's MEG data is typically ~6GB total.
        all_epochs_data = [] # List of numpy arrays
        self.trials = []
        global_idx = 0
        
        for epo_file in subject_files:
            print(f"Reading {epo_file.name}...")
            epochs = mne.read_epochs(str(epo_file), preload=True, verbose='error')
            
            # epochs.get_data() returns numpy array (epochs, channels, times)
            data = epochs.get_data(copy=False).astype(np.float32)
            all_epochs_data.append(data)
            
            # Parse events array [frame, zero, ID]
            events = epochs.events
            
            for i, event_row in enumerate(events):
                event_id = int(event_row[2])
                
                # Check mapping
                if id_to_string_map:
                    image_id = id_to_string_map.get(str(event_id), str(event_id))
                else:
                    image_id = str(event_id)
                    
                self.trials.append({
                    "global_idx": global_idx, # Map back to the concatenated numpy array
                    "image_id": image_id,
                    "event_id": event_id
                })
                global_idx += 1
                
        # Concatenate MEG continuous arrays into single RAM block
        print("Concatenating MEG epochs into RAM...")
        self.meg_data = np.concatenate(all_epochs_data, axis=0)
        self.n_channels, self.seq_len = self.meg_data.shape[1], self.meg_data.shape[2]
        print(f"Preloaded: {self.meg_data.shape} ({self.meg_data.nbytes / 1e6:.0f} MB RAM)")
        
        # Optional: Normalize channels to Z-score across time standard practice
        # self.meg_data = (self.meg_data - self.meg_data.mean(axis=2, keepdims=True)) / (self.meg_data.std(axis=2, keepdims=True) + 1e-6)
        
        # Enforce intersection filtering
        if self.shared_images:
            self.trials = [t for t in self.trials if t["image_id"] in self.shared_images]
            
        # Split logic: Seeded 3-way split by fundamental Image ID to avoid data leak.
        unique_images = sorted(list(set(t["image_id"] for t in self.trials)))
        np.random.seed(42)  # Critical for split alignment across MEG/fMRI
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
            
        print(f"MEG Sub-{subject:02d} | {self.split}: {len(self.trials)} trials | Channels: {self.n_channels}, SeqLen: {self.seq_len}")
        
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
            
        if hasattr(self, 'clip_cache') and image_id in self.clip_cache.files:
            y_clip = torch.tensor(self.clip_cache[image_id], dtype=torch.float32)
        else:
            y_clip = torch.zeros(512, dtype=torch.float32)
            
        return {
            "x": x,
            "image_id": image_id,
            "y_clip": y_clip,
            "meta": trial
        }
