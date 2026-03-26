import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.image_manifest import (
    ensure_shared_conversion_split_lists,
    load_image_split_lists,
    load_named_image_ids,
    resolve_repo_path,
    resolve_shared_manifest_path,
)

class THINGSEEG2Dataset(Dataset):
    """
    Dataset loader for THINGS-EEG2.
    
    Loads preprocessed EEG arrays and their corresponding image concepts.
    Provides the data contract: x (brain tensor), image_id, and y_clip (CLIP embedding).
    """
    
    def __init__(
        self,
        eeg_dir,
        clip_cache_path,
        split="train",
        transform=None,
        subject=1,
        quiet=False,
        shared_only=False,
        shared_manifest_path=None,
        shared_split_dir=None,
        shared_split_seed=42,
        shared_val_concept_count=100,
        shared_test_concept_count=200,
    ):
        self.eeg_dir = Path(eeg_dir)
        self.split = split
        self.transform = transform
        self.subject = subject
        self.quiet = quiet
        self.shared_only = shared_only
        self.shared_split_dir = resolve_repo_path(shared_split_dir) if shared_split_dir else None
        self.shared_split_seed = shared_split_seed
        self.shared_val_concept_count = shared_val_concept_count
        self.shared_test_concept_count = shared_test_concept_count
        
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
            
        if shared_only:
            shared_path = resolve_shared_manifest_path(True, shared_manifest_path)
            shared_images = load_named_image_ids(shared_path)

            if self.shared_split_dir:
                ensure_shared_conversion_split_lists(
                    self.shared_split_dir,
                    shared_images,
                    seed=self.shared_split_seed,
                    val_concept_count=self.shared_val_concept_count,
                    test_concept_count=self.shared_test_concept_count,
                    overwrite=False,
                )
                image_splits = load_image_split_lists(self.shared_split_dir)
                if split not in image_splits:
                    raise ValueError("Split must be 'train', 'val', or 'test'")
                target_images = image_splits[split]
                self.eeg_data, self.concepts, self.files = self._load_shared_split_conditions(
                    eeg_data_dir,
                    shared_images=shared_images,
                    target_images=target_images,
                )
            else:
                eeg_data_path = eeg_data_dir / "preprocessed_eeg_training.npy"
                data_dict = np.load(eeg_data_path, allow_pickle=True).item()
                eeg_data_all = data_dict["preprocessed_eeg_data"] # [16540, Repetitions, Channels, Timepoints]
                concepts_all = self.metadata["train_img_concepts"]
                files_all = self.metadata["train_img_files"]

                shared_indices = [
                    idx for idx, image_file in enumerate(files_all)
                    if Path(image_file).stem in shared_images
                ]
                if not shared_indices:
                    raise ValueError("No EEG training images matched the shared image list")

                np.random.seed(42)
                indices = np.array(shared_indices)
                np.random.shuffle(indices)

                train_end = int(0.8 * len(indices))
                val_end = int(0.9 * len(indices))
                train_idx = indices[:train_end]
                val_idx = indices[train_end:val_end]
                test_idx = indices[val_end:]

                if split == "train":
                    self.eeg_data = eeg_data_all[train_idx]
                    self.concepts = [concepts_all[i] for i in train_idx]
                    self.files = [files_all[i] for i in train_idx]
                elif split == "val":
                    self.eeg_data = eeg_data_all[val_idx]
                    self.concepts = [concepts_all[i] for i in val_idx]
                    self.files = [files_all[i] for i in val_idx]
                elif split == "test":
                    self.eeg_data = eeg_data_all[test_idx]
                    self.concepts = [concepts_all[i] for i in test_idx]
                    self.files = [files_all[i] for i in test_idx]
                else:
                    raise ValueError("Split must be 'train', 'val', or 'test'")

        elif split in ["train", "val"]:
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
        
        if not self.quiet:
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

    def _load_shared_split_conditions(self, eeg_data_dir, shared_images, target_images):
        if not target_images:
            raise ValueError("Shared EEG split manifest selected zero target images")

        blocks = []
        concepts = []
        files = []
        sources = (
            (
                eeg_data_dir / "preprocessed_eeg_training.npy",
                self.metadata["train_img_concepts"],
                self.metadata["train_img_files"],
            ),
            (
                eeg_data_dir / "preprocessed_eeg_test.npy",
                self.metadata["test_img_concepts"],
                self.metadata["test_img_files"],
            ),
        )

        for data_path, source_concepts, source_files in sources:
            data_dict = np.load(data_path, allow_pickle=True).item()
            eeg_data_all = data_dict["preprocessed_eeg_data"]
            selected_indices = [
                idx
                for idx, image_file in enumerate(source_files)
                if Path(image_file).stem in shared_images and Path(image_file).stem in target_images
            ]
            if not selected_indices:
                continue
            blocks.append(eeg_data_all[selected_indices])
            concepts.extend(source_concepts[idx] for idx in selected_indices)
            files.extend(source_files[idx] for idx in selected_indices)

        if not blocks:
            raise ValueError("No EEG conditions matched the requested shared split manifest")

        return np.concatenate(blocks, axis=0), concepts, files
