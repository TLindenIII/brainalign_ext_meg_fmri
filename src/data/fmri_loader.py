from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

from src.data.csv_utils import read_text_table
from src.data.image_manifest import (
    ensure_shared_conversion_split_lists,
    load_image_split_lists,
    load_named_image_ids,
    resolve_repo_path,
    resolve_shared_manifest_path,
)


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
        shared_manifest_path=None,
        shared_split_dir=None,
        shared_split_seed=42,
        shared_val_concept_count=100,
        shared_test_concept_count=200,
        split_mode="official_repeats",
        quiet=False,
    ):
        self.fmri_dir = Path(fmri_dir)
        self.split = split
        self.split_mode = split_mode
        self.effective_split_mode = split_mode
        self.transform = transform
        self.subject = subject
        self.shared_only = shared_only
        self.shared_split_dir = resolve_repo_path(shared_split_dir) if shared_split_dir else None
        self.shared_split_seed = shared_split_seed
        self.shared_val_concept_count = shared_val_concept_count
        self.shared_test_concept_count = shared_test_concept_count
        self.quiet = quiet
        self._log = print if not self.quiet else (lambda *args, **kwargs: None)
        sub_str = f"sub-{subject:02d}"
        
        self._log(f"Loading CLIP cache from {clip_cache_path}")
        self.clip_cache = np.load(clip_cache_path)
        
        self.shared_images = None
        if self.shared_only:
            shared_path = resolve_shared_manifest_path(True, shared_manifest_path)
            self.shared_images = load_named_image_ids(shared_path)
            self._log(f"Loaded {len(self.shared_images)} shared images for cross-modal filtering")
                
        # ---- Load the stimulus metadata TSV ----
        beta_dir = self.fmri_dir / "derivatives" / "ICA-betas" / sub_str / "voxel-metadata"
        metadata_tsv = beta_dir / f"{sub_str}_task-things_stimulus-metadata.tsv"
        
        if not metadata_tsv.exists():
            raise FileNotFoundError(f"Stimulus metadata not found: {metadata_tsv}")
            
        df = read_text_table(metadata_tsv, expected_columns={"stimulus"})
        self._log(f"Loaded {len(df)} trials from {metadata_tsv.name}")

        # ---- Load HDF5 metadata first so split/feature stats use only train trials ----
        h5_path = beta_dir / f"{sub_str}_task-things_voxel-wise-responses.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Voxel data not found: {h5_path}")

        self._log(f"Loading voxel data from {h5_path.name}...")
        with h5py.File(str(h5_path), 'r') as f:
            full_data = f['ResponseData']['block0_values']  # (n_voxels, n_trials)
            n_voxels_full, n_trials_h5 = full_data.shape
            self._log(f"Full HDF5: {n_voxels_full} voxels × {n_trials_h5} trials")

            # ---- Build trial list before feature selection so stats use train only ----
            all_trials = []
            for row_idx, row in df.iterrows():
                if row_idx >= n_trials_h5:
                    break

                stim = str(row.get('stimulus', ''))
                if stim == 'nan' or not stim:
                    continue

                image_id = Path(stim).stem
                if self.shared_images and image_id not in self.shared_images:
                    continue

                trial_type = str(row.get("trial_type", "")).strip().lower()

                all_trials.append(
                    {
                        "trial_idx": row_idx,
                        "subject": sub_str,
                        "image_id": image_id,
                        "trial_type": trial_type,
                    }
                )

            train_trial_indices = self._assign_splits(all_trials)
            if train_trial_indices.size == 0:
                raise ValueError("No training trials available for fMRI feature selection/statistics")

            # Select top-K voxels using only training trials to avoid leakage.
            self._log(
                f"Computing per-voxel variance on {train_trial_indices.size} train trials "
                f"for top-{N_TOP_VOXELS} selection..."
            )
            n_sample = min(3000, train_trial_indices.size)
            sample_idx = np.linspace(0, train_trial_indices.size - 1, n_sample, dtype=int)
            variance_trial_indices = train_trial_indices[sample_idx]
            sample_data = full_data[:, variance_trial_indices]
            voxel_var = np.var(sample_data, axis=1)

            top_k_indices = np.argsort(voxel_var)[-N_TOP_VOXELS:]
            top_k_indices.sort()

            var_retained = voxel_var[top_k_indices].sum() / voxel_var.sum() * 100
            self._log(f"Selected {N_TOP_VOXELS} voxels ({var_retained:.1f}% of train variance)")

            self._log("Preloading selected voxels into RAM...")
            self.voxel_data = full_data[top_k_indices, :].T.astype(np.float32)

        self.n_voxels = N_TOP_VOXELS
        self._log(f"Preloaded: {self.voxel_data.shape} ({self.voxel_data.nbytes / 1e6:.0f} MB in RAM)")

        # Compute normalization statistics from the training split only.
        train_selected = self.voxel_data[train_trial_indices]
        self.voxel_mean = train_selected.mean(axis=0)
        self.voxel_std = train_selected.std(axis=0)
        self.voxel_std[self.voxel_std < 1e-6] = 1.0

        if self.split == "train":
            self.trials = [t for t in all_trials if t["assigned_split"] == "train"]
        elif self.split == "val":
            self.trials = [t for t in all_trials if t["assigned_split"] == "val"]
        elif self.split == "test":
            self.trials = [t for t in all_trials if t["assigned_split"] == "test"]
        elif self.split == "all":
            self.trials = list(all_trials)
        else:
            raise ValueError("Split must be 'train', 'val', 'test', or 'all'")

        self._log(
            f"fMRI {sub_str} | {self.split} ({self.effective_split_mode}): {len(self.trials)} trials | "
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

    def _assign_splits(self, all_trials):
        if not all_trials:
            raise ValueError(
                "No fMRI image IDs remain after preprocessing/filtering. "
                "Check shared-image settings and stimulus metadata."
            )

        if self.shared_only and self.shared_split_dir:
            return self._assign_shared_conversion_splits(all_trials)
        if self.split_mode == "official_repeats":
            return self._assign_official_repeat_splits(all_trials)
        if self.split_mode == "random_strict":
            return self._assign_random_strict_splits(all_trials)

        raise ValueError(
            f"Unsupported fMRI split mode '{self.split_mode}'. "
            "Expected 'official_repeats' or 'random_strict'."
        )

    def _assign_official_repeat_splits(self, all_trials):
        official_train_trials = [trial for trial in all_trials if trial["trial_type"] == "train"]
        official_test_trials = [trial for trial in all_trials if trial["trial_type"] == "test"]
        if not official_train_trials or not official_test_trials:
            if self.shared_only:
                available_groups = []
                if official_train_trials:
                    available_groups.append("train")
                if official_test_trials:
                    available_groups.append("test")
                available_text = ", ".join(available_groups) if available_groups else "none"
                self._log(
                    "Shared-manifest filtering removed one side of the official fMRI split "
                    f"(available trial_type groups: {available_text}). "
                    "Falling back to deterministic random_strict splitting for this shared subset."
                )
                self.effective_split_mode = "random_strict_fallback"
                return self._assign_random_strict_splits(all_trials)
            raise ValueError(
                "official_repeats requires metadata 'trial_type' values for both train and test rows"
            )

        official_train_images = sorted({trial["image_id"] for trial in official_train_trials})
        rng = np.random.RandomState(42)
        shuffled = official_train_images.copy()
        rng.shuffle(shuffled)

        val_count = max(1, int(0.1 * len(shuffled)))
        train_images = set(shuffled[:-val_count])
        val_images = set(shuffled[-val_count:])
        test_images = {trial["image_id"] for trial in official_test_trials}

        self.image_splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images,
        }

        for trial in all_trials:
            if trial["trial_type"] == "test":
                trial["assigned_split"] = "test"
            elif trial["image_id"] in val_images:
                trial["assigned_split"] = "val"
            else:
                trial["assigned_split"] = "train"

        return np.array(
            [trial["trial_idx"] for trial in all_trials if trial["assigned_split"] == "train"],
            dtype=int,
        )

    def _assign_random_strict_splits(self, all_trials):
        unique_images = sorted({trial["image_id"] for trial in all_trials})
        if not unique_images:
            raise ValueError(
                "No fMRI image IDs remain after preprocessing/filtering. "
                "Check shared-image settings and stimulus metadata."
            )

        rng = np.random.RandomState(42)
        shuffled = unique_images.copy()
        rng.shuffle(shuffled)

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

        for trial in all_trials:
            if trial["image_id"] in train_images:
                trial["assigned_split"] = "train"
            elif trial["image_id"] in val_images:
                trial["assigned_split"] = "val"
            else:
                trial["assigned_split"] = "test"

        return np.array(
            [trial["trial_idx"] for trial in all_trials if trial["assigned_split"] == "train"],
            dtype=int,
        )

    def _assign_shared_conversion_splits(self, all_trials):
        ensure_shared_conversion_split_lists(
            self.shared_split_dir,
            self.shared_images or {trial["image_id"] for trial in all_trials},
            seed=self.shared_split_seed,
            val_concept_count=self.shared_val_concept_count,
            test_concept_count=self.shared_test_concept_count,
            overwrite=False,
        )
        self.image_splits = load_image_split_lists(self.shared_split_dir)
        excluded_path = self.shared_split_dir / "excluded.txt"
        excluded_images = load_named_image_ids(excluded_path) if excluded_path.exists() else set()

        filtered_trials = []
        for trial in all_trials:
            image_id = trial["image_id"]
            if image_id in self.image_splits["train"]:
                trial["assigned_split"] = "train"
                filtered_trials.append(trial)
            elif image_id in self.image_splits["val"]:
                trial["assigned_split"] = "val"
                filtered_trials.append(trial)
            elif image_id in self.image_splits["test"]:
                trial["assigned_split"] = "test"
                filtered_trials.append(trial)
            elif image_id in excluded_images:
                continue

        all_trials[:] = filtered_trials
        if not all_trials:
            raise ValueError("No fMRI trials matched the requested shared conversion split manifest")

        self.effective_split_mode = "shared_conversion_manifest"
        return np.array(
            [trial["trial_idx"] for trial in all_trials if trial["assigned_split"] == "train"],
            dtype=int,
        )
