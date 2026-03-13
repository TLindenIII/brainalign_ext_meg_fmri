import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml

class THINGSEEG2Dataset(Dataset):
    """
    Dataset loader for THINGS-EEG2.
    
    Loads preprocessed EEG arrays and their corresponding image concepts.
    Provides the data contract: x (brain tensor), image_id, and y_clip (CLIP embedding).
    """
    
    def __init__(self, eeg_dir, clip_cache_path, split="train", transform=None, subject=1, quiet=False):
        self.eeg_dir = Path(eeg_dir)
        self.split = split
        self.transform = transform
        self.subject = subject
        self.quiet = quiet
        
        # Load CLIP embeddings
        if not self.quiet:
            print(f"Loading CLIP cache from {clip_cache_path}")
        self.clip_cache = np.load(clip_cache_path)
        
        # Load metadata
        metadata_path = self.eeg_dir / "stimuli" / "image_metadata.npy"
        self.metadata = np.load(metadata_path, allow_pickle=True).item()
        
        if not self.quiet:
            print(f"Loading EEG {split} data for subject {subject:02d}...")
        
        eeg_data_dir = self.eeg_dir / "preprocessed" / f"sub-{self.subject:02d}"
        if not eeg_data_dir.exists():
            raise ValueError(f"Subject directory not found: {eeg_data_dir}")
            
        if split in ["train", "val"]:
            eeg_data_path = eeg_data_dir / "preprocessed_eeg_training.npy"
            data_dict = np.load(eeg_data_path, allow_pickle=True).item()
            eeg_data_all = data_dict["preprocessed_eeg_data"] # [16540, Repetitions, Channels, Timepoints]
            
            # Metadata keys
            concepts_all = self.metadata["train_img_concepts"]
            files_all = self.metadata["train_img_files"]
            
            # Deterministic split (80/10/10 approx = 15800/740/200 paper baseline)
            np.random.seed(42)
            indices = np.arange(len(files_all))
            np.random.shuffle(indices)
            
            train_idx = indices[:-740]
            val_idx = indices[-740:]
            
            if split == "train":
                self.eeg_data = eeg_data_all[train_idx]
                self.concepts = [concepts_all[i] for i in train_idx]
                self.files = [files_all[i] for i in train_idx]
            else: # val
                self.eeg_data = eeg_data_all[val_idx]
                self.concepts = [concepts_all[i] for i in val_idx]
                self.files = [files_all[i] for i in val_idx]
            
            
        elif split == "test":
            eeg_data_path = eeg_data_dir / "preprocessed_eeg_test.npy"
            data_dict = np.load(eeg_data_path, allow_pickle=True).item()
            self.eeg_data = data_dict["preprocessed_eeg_data"]
            self.concepts = self.metadata["test_img_concepts"]
            self.files = self.metadata["test_img_files"]
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")
            
        # Flatten the conditions and repetitions into single trials
        self.n_conditions, self.n_reps, self.n_channels, self.n_time = self.eeg_data.shape
        self.n_total_trials = self.n_conditions * self.n_reps
        
        print(f"Loaded {self.n_conditions} conditions with {self.n_reps} repetitions each.")
        
    def __len__(self):
        return self.n_total_trials
        
    def __getitem__(self, idx):
        # Calculate condition and repetition indices
        cond_idx = idx // self.n_reps
        rep_idx = idx % self.n_reps
        
        # Extract brain data: shape [Channels, Timepoints]
        x = self.eeg_data[cond_idx, rep_idx, :, :]
        x = torch.tensor(x, dtype=torch.float32)
        
        if self.transform:
            x = self.transform(x)
            
        # Extract image_id
        # The filename serves as our image_id (e.g., 'aardvark_01b.jpg')
        # So we strip out the folder path
        image_file_path = self.files[cond_idx]
        image_id = Path(image_file_path).stem
        
        # Extract CLIP embedding
        if image_id in self.clip_cache.files:
            y_clip = torch.tensor(self.clip_cache[image_id], dtype=torch.float32)
        else:
            # Fallback (should not happen if cache built correctly)
            print(f"Warning: {image_id} not found in CLIP cache!")
            y_clip = torch.zeros(512, dtype=torch.float32) 
            
        return {
            "x": x,
            "image_id": image_id,
            "y_clip": y_clip,
            "meta": {"cond_idx": cond_idx, "rep_idx": rep_idx}
        }
