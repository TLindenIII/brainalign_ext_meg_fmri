from pathlib import Path
import os
import sys

import h5py
import mne
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.eeg_loader import THINGSEEG2Dataset
from src.data.fmri_loader import N_TOP_VOXELS


def load_config():
    with open(ROOT / "config.yaml", "r") as handle:
        return yaml.safe_load(handle)


def clip_shape(clip_cache_path):
    clip_cache = np.load(clip_cache_path)
    first_key = clip_cache.files[0]
    return clip_cache[first_key].shape


def inspect_eeg(config, clip_cache_path):
    print("--- EEG ---")
    eeg = THINGSEEG2Dataset(config["data"]["eeg_dir"], clip_cache_path, split="train", quiet=True)
    sample = eeg[0]
    print(f"Size: {len(eeg)}, x: {sample['x'].shape}, y: {sample['y_clip'].shape}")


def inspect_meg(config, clip_cache_path, subject=1):
    print("--- MEG ---")
    prep_dir = ROOT / config["data"]["meg_dir"] / "derivatives" / "preprocessed"
    merged_file = prep_dir / f"preprocessed_P{subject}-epo.fif"
    if merged_file.exists():
        epoch_file = merged_file
    else:
        matches = sorted(prep_dir.glob(f"preprocessed_P{subject}-epo-*.fif"))
        if not matches:
            raise FileNotFoundError(f"No MEG preprocessed file found for subject {subject} in {prep_dir}")
        epoch_file = matches[0]

    epochs = mne.read_epochs(str(epoch_file), preload=False, verbose="error")
    tmin = max(float(epochs.times[0]), -0.5)
    tmax = min(float(epochs.times[-1]), 1.0)
    seq_len = int(round((tmax - tmin) * 120.0)) + 1
    print(
        f"File: {epoch_file.name}, estimated x: ({epochs.info['nchan']}, {seq_len}), "
        f"y: {clip_shape(clip_cache_path)}"
    )


def inspect_fmri(config, clip_cache_path):
    print("--- fMRI ---")
    subject = 1
    sub_str = f"sub-{subject:02d}"
    voxel_path = (
        ROOT
        / config["data"]["fmri_dir"]
        / "derivatives"
        / "ICA-betas"
        / sub_str
        / "voxel-metadata"
        / f"{sub_str}_task-things_voxel-wise-responses.h5"
    )
    with h5py.File(voxel_path, "r") as handle:
        n_voxels_full, n_trials = handle["ResponseData"]["block0_values"].shape

    print(
        f"File: {voxel_path.name}, trials: {n_trials}, estimated x: ({min(N_TOP_VOXELS, n_voxels_full)},), "
        f"y: {clip_shape(clip_cache_path)}"
    )


if __name__ == "__main__":
    config = load_config()
    clip_cache_path = os.path.join(config["data"]["clip_cache_dir"], "ViT-B-32.npz")
    inspect_eeg(config, clip_cache_path)
    inspect_meg(config, clip_cache_path)
    inspect_fmri(config, clip_cache_path)
