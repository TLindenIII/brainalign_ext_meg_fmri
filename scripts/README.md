# Scripts Directory

This directory contains the user-facing entry points for data prep, training, and evaluation. The project now uses one CLIP space across EEG, MEG, and fMRI, with manifest files controlling each modality's native image set and the pairwise or 3-way intersections used for conversion.

## Core Workflow

### Data-prep scripts

- `scripts/build_image_manifests.py`
  - Writes modality manifests under `data/manifests/`.
  - Always produces:
    - `data/manifests/eeg_all.tsv`
    - `data/manifests/fmri_all.tsv`
    - `data/manifests/meg_numeric.tsv`
  - If a THINGS image-number map is available, also produces:
    - `data/manifests/meg_all.tsv`
    - `data/manifests/all_modalities_union.tsv`
    - `data/manifests/intersections/eeg_meg.txt`
    - `data/manifests/intersections/eeg_fmri.txt`
    - `data/manifests/intersections/fmri_meg.txt`
    - `data/manifests/intersections/eeg_fmri_meg.txt`
- `scripts/build_shared_images.py`
  - Compatibility wrapper around `scripts/build_image_manifests.py`.
  - Keeps the legacy `data/shared_images.txt` file in sync from `fmri_meg.txt` when that intersection exists.
- `scripts/build_clip_cache.py`
  - Preferred entry point for CLIP cache creation.
  - Use `--manifest data/manifests/all_modalities_union.tsv --image-root <THINGS root>` to build one cache for all modalities.

### Training scripts

- `scripts/train_all_subjects.sh`
- `scripts/train_all_subjects.ps1`

These discover the locally available subjects for a modality and launch `src/train.py` sequentially.

Key behavior:

- Auto-detects `.venv` first, then `venv`, then falls back to `python`.
- Skips subjects with existing `*_best.pt` checkpoints unless `--resume` / `-Resume` is used.
- Accepts `--shared-only` / `-SharedOnly` for shared-manifest training.
- Accepts `--shared-manifest <path>` / `-SharedManifest <path>` to target a specific intersection file.

Checkpoint naming:

- EEG: `checkpoints/eeg/eeg_brainalign_sub01_best.pt`
- fMRI: `checkpoints/fmri/fmri_brainalign_sub01_best.pt`
- MEG: `checkpoints/meg/meg_brainalign_sub01_attnpool_best.pt`
- Shared-only runs add `_shared` before `_best.pt` / `_latest.pt`

### Evaluation scripts

- `scripts/evaluate_retrieval.py`
  - Reports modality-to-image and image-to-modality Top-1, Top-5, and CLIP 2-way.
- `scripts/evaluate_conversion.py`
  - Reports modality-to-modality conversion metrics on a shared manifest.
  - Defaults to `data/manifests/intersections/<modalities>.txt` when present.
- `scripts/evaluate_conversion_matrix.py`
  - Evaluates all subject-pair conversions for two modalities while loading each subject only once.
  - Writes the same per-pair `results/conversion/*.txt` files as `scripts/evaluate_conversion.py`.
- `scripts/evaluate_eeg_table.py`
  - EEG-only summary table generator.
- `scripts/summarize_results.py`
  - Aggregates retrieval and conversion `.txt` outputs into CSV summaries, a markdown report, and paper-style decoding/conversion tables under `results/summary/`.

## Requirements For Correct MEG Training

MEG no longer falls back to EEG stimulus metadata. To train MEG correctly, the repo now expects a THINGS image-number map at `data/things_image_map.tsv` or the path configured by `data.things_image_map_path` in `config.yaml`.

Minimum columns:

- `image_number`
- `image_id`

Recommended additional column:

- `relative_path`

If the repo contains the THINGS OSF archive metadata at `osfstorage-archive/01_image-level/image-paths.csv`, `scripts/build_image_manifests.py` will auto-generate `data/things_image_map.tsv`.

If neither a map nor the OSF image-path metadata is present, the manifest builder will generate `data/manifests/things_image_map.template.tsv`, and MEG training/evaluation will stop with a clear error instead of silently using the wrong image space.

## Helper / Sanity Scripts

- `scripts/test_model.py`
- `scripts/print_shapes.py`
- `scripts/print_fmri_struct.py`
- `scripts/test_meg_metadata.py`
- `scripts/test_pd.py`
- `scripts/inspect_weights.py`
- `scripts/patch_notebook.py`

## Runbook

For the ordered training-PC workflow, see [scripts/RUNBOOK.Rmd](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/scripts/RUNBOOK.Rmd).
