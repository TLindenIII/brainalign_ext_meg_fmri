# Scripts Directory

This directory contains the user-facing scripts for training, evaluation, and lightweight sanity checks in the BrainAlign project. Model and loader implementation still live in `src/`, but the commands you run are centralized here.

## Available Scripts

### `train_all_subjects.sh` (Linux/Mac) & `train_all_subjects.ps1` (Windows)

These scripts automate the process of sequentially running the training loop (`src/train.py`) across all locally available subjects for a specified modality. They ensure that the model is trained consistently and provide a streamlined way to run the dataset without manual intervention for each subject.

#### Features:

- **Sequential Execution:** Discovers and loops over the locally available subjects for the requested modality.
- **Checkpoint Skipping:** If it detects an existing `best.pt` checkpoint for a subject, it will skip training for that subject to save time, unless forced or resuming.
- **Virtual Environment:** Automatically sources `.venv/bin/activate` (`.sh`) or `.\venv\Scripts\Activate.ps1` (`.ps1`) if it exists in the root directory.
- **Shared-Subset Training:** Optional `--shared-only` / `-SharedOnly` flag to train MEG/fMRI only on the `shared_images.txt` intersection used for conversion.
- **Final Evaluation:** EEG runs still auto-generate the EEG summary table; MEG/fMRI runs now print the recommended `scripts/evaluate_retrieval.py` / `scripts/evaluate_conversion.py` follow-up commands instead of calling the EEG-only summary script.

#### Checkpoint System:

By default, `train.py` creates two kinds of comprehensive state dictionaries for every subject during training:

1. `*_best.pt`: Saved **only** when the model achieves a new high-score on the isolated Validation set. Resuming from this file restores the model to its absolute smartest observed state (Early Stopping).
2. `*_latest.pt`: Saved indiscriminately at the exact end of every single epoch. Resuming from this file restores the exact momentum of your last trained epoch, regardless of whether the model had begun overfitting.

**Storage Location & Modality Isolation:**
Checkpoints are rigidly isolated by modality to prevent cross-contamination. They are saved in the root `checkpoints/` directory following this structure:

- `checkpoints/eeg/eeg_brainalign_sub01_best.pt`
- `checkpoints/fmri/fmri_brainalign_sub01_latest.pt`
- `checkpoints/meg/meg_brainalign_sub01_best.pt`

If `--shared-only` is used, the scripts add a `_shared` suffix before `_best.pt` / `_latest.pt` so full-data retrieval models and shared-subset conversion models do not overwrite each other.

#### Usage:

Run the appropriate script for your OS from the **root directory** of the project:

**Linux/Mac:**
```bash
./scripts/train_all_subjects.sh [OPTIONS]
```

**Windows (PowerShell):**
```powershell
.\scripts\train_all_subjects.ps1 -Modality [OPTIONS]
```

#### Arguments:

- `--modality <type>` : (Required-ish, defaults to `eeg`) The data modality you want to train on. Acceptable values are `eeg`, `meg`, or `fmri`.
- `--epochs <number>` : (Optional) Overrides the default number of epochs set in `config.yaml` (Defaults: `eeg`=60, `meg`=60, `fmri`=500). Example: `--epochs 200`.
- `--resume` : (Optional) Tells the script to look for a `latest.pt` (or `best.pt` fallback) comprehensive state dictionary to resume training exactly where it left off, rather than starting from Epoch 1 or skipping the subject entirely.
- `--shared-only` / `-SharedOnly` : (Optional) Restricts MEG/fMRI training to the shared image intersection used for cross-modal conversion and writes checkpoints with a `_shared` suffix.

#### Examples:

**Train EEG (uses config.yaml default of 60 epochs):**

```bash
./scripts/train_all_subjects.sh
```

**Train fMRI (uses config.yaml default of 500 epochs):**

```bash
./scripts/train_all_subjects.sh --modality fmri
```

**Resume an interrupted fMRI training run:**

```bash
./scripts/train_all_subjects.sh --modality fmri --resume
```

**Run in the background (helpful for remote servers or long runs):**

```bash
nohup ./scripts/train_all_subjects.sh --modality fmri --resume &> full_train_out.log &
```

**Monitor the background training logs:**

```bash
tail -f full_train_out.log
```

### Helper / Sanity Scripts

These lightweight helper scripts are also centralized in this folder:

- `scripts/test_model.py` : Instantiates `BrainAlignModel` and verifies the forward output shape.
- `scripts/print_shapes.py` : Prints one sample shape for EEG, MEG, and fMRI loaders.
- `scripts/print_fmri_struct.py` : Prints one sample fMRI metadata record.
- `scripts/test_meg_metadata.py` : Dumps key EEG image metadata arrays used for MEG mapping.
- `scripts/test_pd.py` : Quick pandas smoke test for reading MEG `events.tsv` files.
- `scripts/inspect_weights.py` : Lists the CBraMod pretrained weight tensor shapes.
- `scripts/patch_notebook.py` : Applies the retrieval notebook cell patch in-place.

### Evaluation / Data-Prep Scripts

- `scripts/build_shared_images.py` : Rebuilds `data/shared_images.txt` by intersecting valid MEG stimulus IDs with fMRI stimulus metadata.
- `scripts/evaluate_retrieval.py` : Runs bidirectional retrieval metrics for EEG, MEG, or fMRI checkpoints.
- `scripts/evaluate_conversion.py` : Runs shared-image modality-conversion metrics between any two trained modalities.
- `scripts/evaluate_eeg_table.py` : Generates the EEG summary tables from the saved EEG checkpoints.

### Runbook

For the end-to-end order of operations on the training PC, see:

- `scripts/RUNBOOK.Rmd`
