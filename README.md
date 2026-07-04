# BrainAlign Extension on THINGS

This repository trains subject-specific EEG, MEG, and fMRI encoders into a shared frozen CLIP image space, then evaluates:

- modality-to-image retrieval
- image-to-modality retrieval
- cross-modality conversion on manifest-backed shared image pools
- isolated experimental objectives under separate checkpoint and result namespaces

## Start Here

Use these tracked sources first:

- [scripts/README.md](scripts/README.md): script inventory and entry-point behavior
- [scripts/RUNBOOK.Rmd](scripts/RUNBOOK.Rmd): ordered execution guide
- `results/summary/`: generated baseline summaries
- `results/experiments/*/summary/`: generated summaries from completed runs

## Current Implementation

- The repo currently uses one frozen CLIP target space: `ViT-B-32.npz`.
- The CLIP cache is generated locally under `clip_cache/` and is not tracked.
- Models are trained per subject.
- Full retrieval and shared-pool conversion are separate benchmarks with separate checkpoints.
- EEG uses a CBraMod-backed BrainAlign-style encoder.
- MEG uses a dedicated temporal CNN with attention pooling.
- fMRI uses a residual MLP over variance-selected voxels.
- Experimental objectives are isolated under `checkpoints/experiments/<name>/` and `results/experiments/<name>/`.

Current maintained conclusions:

- Full retrieval is strong in all three modalities under the current CLIP benchmark.
- Pairwise and three-way shared-pool conversion are stricter and should be compared only within matched scope.
- Experiment 1 isolates the asymmetric brain-to-CLIP objective on EEG/MEG.
- Experiment 2 adds explicit paired cross-modal pressure on EEG/MEG and is mixed.
- Experiment 3 jointly trains EEG, MEG, and fMRI on the matched three-way pool and improves raw three-way conversion over the matched baseline in all six directions.

## Dataset Acquisition

This repo does not redistribute the raw datasets. You need local copies of:

- **THINGS image database**: OSF project `jum2f`
- **THINGS-EEG2**: OSF project `3jk45`
- **THINGS-MEG**: OpenNeuro `ds004212`
- **THINGS-fMRI**: OpenNeuro `ds004192`

Recommended sources:

- THINGS initiative dataset index: `https://things-initiative.org/`
- THINGS image database (OSF): `https://osf.io/jum2f/`
- THINGS-EEG2 (OSF): `https://osf.io/3jk45/`
- THINGS-MEG (OpenNeuro): `https://openneuro.org/datasets/ds004212`
- THINGS-fMRI (OpenNeuro): `https://openneuro.org/datasets/ds004192`

Example commands:

```bash
pip install osfclient datalad

# THINGS image database
osf -p jum2f clone THINGS-database

# THINGS-EEG2
osf -p 3jk45 clone THINGS-EEG2

# THINGS-MEG
datalad clone https://github.com/OpenNeuroDatasets/ds004212.git data/things-meg-ds004212
cd data/things-meg-ds004212 && datalad get .

# THINGS-fMRI
datalad clone https://github.com/OpenNeuroDatasets/ds004192.git data/things-fmri-ds004192
cd data/things-fmri-ds004192 && datalad get .
```

After download, update `config.yaml` if your local paths differ from the default
repo-relative layout.

## Quick Workflow

Build manifests:

```bash
./.venv/bin/python scripts/build_image_manifests.py --config config.yaml
```

Build the local CLIP cache before training or evaluation:

```bash
./.venv/bin/python scripts/build_clip_cache.py \
  --config config.yaml \
  --manifest data/manifests/all_modalities_union.tsv \
  --image-root <THINGS image root>
```

Train baseline full retrieval models:

```bash
./scripts/train_all_subjects.sh --modality eeg --resume
./scripts/train_all_subjects.sh --modality meg --resume
./scripts/train_all_subjects.sh --modality fmri --resume
```

Evaluate baseline retrieval and shared conversion suites:

```bash
./.venv/bin/python -m scripts.evaluate_all --modalities eeg,meg,fmri --skip-shared-suite --clean
./.venv/bin/python -m scripts.evaluate_all --modalities eeg,meg,fmri --skip-full-retrieval --shared-manifest data/manifests/conversion_pools/eeg_fmri_meg.txt --clean
```

Rebuild summaries:

```bash
./.venv/bin/python -m scripts.summarize_results --results-root results --output-dir results/summary
```

For the full tracked workflow and experiment commands, use [scripts/README.md](scripts/README.md) and the generated summaries under `results/`.

## Documentation Roles

- `README.md`: front door only
- `scripts/README.md`: script reference
- `scripts/RUNBOOK.Rmd`: ordered execution guide
- `notebooks/*.md`: historical notebook companions, not canonical docs
- `results/**/summary_report.md`: generated outputs, not hand-maintained docs
- `src/vendored/**/README.md`: upstream vendor docs
- `non_paper/` (ignored locally): internal design notes, report drafts, references, and presentation materials
