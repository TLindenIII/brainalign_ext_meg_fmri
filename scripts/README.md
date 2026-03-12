# Scripts Directory

This directory contains shell scripts for automating the training and evaluation pipeline of the BrainAlign project across multiple subjects.

## Available Scripts

### `train_all_subjects.sh` (Linux/Mac) & `train_all_subjects.ps1` (Windows)

These scripts automate the process of sequentially running the training loop (`src/train.py`) across all 10 subjects for a specified modality. They ensure that the model is trained consistently and provides a streamlined way to run the entire dataset without manual intervention for each subject.

#### Features:

- **Sequential Execution:** Loops from Subject 1 to 10 automatically.
- **Checkpoint Skipping:** If it detects an existing `best.pt` checkpoint for a subject, it will skip training for that subject to save time, unless forced or resuming.
- **Virtual Environment:** Automatically sources `.venv/bin/activate` (`.sh`) or `.\venv\Scripts\Activate.ps1` (`.ps1`) if it exists in the root directory.
- **Final Evaluation:** Once all 10 subjects are completed, it automatically calls `src/evaluate_table.py` (`-m src.evaluate_table` on Windows) to generate the final averaged SOTA performance matrix.

#### Checkpoint System:

By default, `train.py` creates two kinds of comprehensive state dictionaries for every subject during training:

1. `*_best.pt`: Saved **only** when the model achieves a new high-score on the isolated Validation set. Resuming from this file restores the model to its absolute smartest observed state (Early Stopping).
2. `*_latest.pt`: Saved indiscriminately at the exact end of every single epoch. Resuming from this file restores the exact momentum of your last trained epoch, regardless of whether the model had begun overfitting.

**Storage Location & Modality Isolation:**
Checkpoints are rigidly isolated by modality to prevent cross-contamination. They are saved in the root `checkpoints/` directory following this structure:

- `checkpoints/eeg/eeg_brainalign_sub01_best.pt`
- `checkpoints/fmri/fmri_brainalign_sub01_latest.pt`
- `checkpoints/meg/meg_brainalign_sub01_best.pt`

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
