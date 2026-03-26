# BrainAlign Extension on THINGS

This repository trains modality-specific brain encoders that align EEG, MEG, and fMRI responses to a shared CLIP image space, then evaluates:

- full-dataset retrieval for each modality
- pairwise modality conversion on shared image pools
- later, a stricter 3-way shared conversion protocol for paper-quality comparisons

The repo now treats retrieval and conversion as separate benchmarks with separate checkpoints and explicit split manifests.

## Current Scope

- EEG full retrieval uses the THINGS-EEG2 official train/test release.
- fMRI full retrieval uses the THINGS-fMRI official `trial_type=train/test` split.
- MEG full retrieval uses a fixed EEG-style held-out split generated under [data/manifests/splits/meg/fixed_image_holdout](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/data/manifests/splits/meg/fixed_image_holdout).
- Shared conversion uses dedicated manifest-backed pools under [data/manifests/conversion_pools](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/data/manifests/conversion_pools) and shared split manifests under [data/manifests/splits/conversion](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/data/manifests/splits/conversion).

## Full Retrieval Protocol

Retrieval is always scored at the `image_id` level after averaging repeated trials for the same image. In the repo tables, “class” means candidate `image_id`, not semantic concept.

| Modality | Split Source | Train Images | Val Images | Test Images | Train Concepts | Val Concepts | Test Concepts | Test Novel Images? | Test Novel Concepts? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| EEG | Official EEG train + official EEG test | 15,800 | 740 | 200 | 1,654 | 599 | 200 | Yes | Yes |
| MEG | Fixed manifest-backed holdout | 18,023 | 2,003 | 200 | 1,654 | 1,199 | 200 | Yes | Yes |
| fMRI | Official `trial_type=train/test` | 7,776 | 864 | 100 | 720 | 499 | 100 | Yes | No |

Notes:

- EEG full retrieval uses the official THINGS-EEG2 train pool for `train/val`, and the official EEG test pool for `test`.
- MEG full retrieval uses one held-out image from each of 200 held-out concepts for test and excludes the remaining images from those 200 concepts from training.
- fMRI full retrieval uses the official test images, but their concepts are already present in official train.
- Candidate set sizes for the current full retrieval benchmark are therefore:
  - EEG: `200`
  - MEG: `200`
  - fMRI: `100`

## Shared Conversion Protocol

Conversion models are separate from full retrieval models. They do not reuse the modality-native retrieval test setup.

Each conversion pool is built from images that are eligible for training in every modality in that pool:

- EEG contributes official training images only.
- fMRI contributes official `trial_type=train` images only.
- MEG contributes its full available image universe.

Each pool then gets one deterministic concept-holdout split:

- `train`: all images from the remaining train concepts
- `val`: 1 image from each of 100 held-out concepts
- `test`: 1 image from each of 200 held-out concepts
- `excluded`: the remaining images from those 300 held-out `val/test` concepts

Evaluation is still exact-image retrieval, not concept-level scoring. The held-out concepts just make the split consistent across modalities and directions.

This is the protocol that makes all 6 pairwise conversion directions comparable.

| Pool | Raw Overlap Images / Concepts | Trainable Pool Images / Concepts | Train Images / Concepts | Val Images / Concepts | Test Images / Concepts | Excluded Images |
| --- | --- | --- | --- | --- | --- | ---: |
| EEG↔MEG | `16,718 / 1,832` | `16,540 / 1,654` | `13,540 / 1,354` | `100 / 100` | `200 / 200` | 2,700 |
| EEG↔fMRI | `6,407 / 710` | `6,330 / 633` | `3,330 / 333` | `100 / 100` | `200 / 200` | 2,700 |
| MEG↔fMRI | `8,740 / 720` | `8,640 / 720` | `5,040 / 420` | `100 / 100` | `200 / 200` | 3,300 |
| EEG↔MEG↔fMRI | `6,407 / 710` | `6,330 / 633` | `3,330 / 333` | `100 / 100` | `200 / 200` | 2,700 |

Important distinctions:

- Raw overlaps live under [data/manifests/intersections](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/data/manifests/intersections).
- Trainable conversion pools live under [data/manifests/conversion_pools](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/data/manifests/conversion_pools).
- Shared conversion training and evaluation should use the `conversion_pools` manifests, not the raw `intersections` manifests.

## Checkpoints

Full retrieval checkpoints:

- EEG: [checkpoints/eeg](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/checkpoints/eeg)
- MEG: [checkpoints/meg](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/checkpoints/meg)
- fMRI: [checkpoints/fmri](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/checkpoints/fmri)

Shared conversion checkpoints:

- EEG↔MEG: [checkpoints/conversion/shared-eeg-meg](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/checkpoints/conversion/shared-eeg-meg)
- EEG↔fMRI: [checkpoints/conversion/shared-eeg-fmri](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/checkpoints/conversion/shared-eeg-fmri)
- MEG↔fMRI: [checkpoints/conversion/shared-fmri-meg](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/checkpoints/conversion/shared-fmri-meg)
- EEG↔MEG↔fMRI: [checkpoints/conversion/shared-eeg-fmri-meg](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/checkpoints/conversion/shared-eeg-fmri-meg)

MEG full and shared checkpoints use the `_temporalcnn` stem.

## Build Manifests

Rebuild manifests and split files after changing protocol code:

```bash
./.venv/bin/python scripts/build_image_manifests.py --config config.yaml
```

This writes:

- modality manifests under [data/manifests](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/data/manifests)
- raw overlap manifests under [data/manifests/intersections](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/data/manifests/intersections)
- conversion-pool manifests under [data/manifests/conversion_pools](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/data/manifests/conversion_pools)
- MEG full retrieval split manifests under [data/manifests/splits/meg/fixed_image_holdout](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/data/manifests/splits/meg/fixed_image_holdout)
- pairwise and 3-way conversion split manifests under [data/manifests/splits/conversion](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/data/manifests/splits/conversion)

## Training

Full retrieval models:

```bash
./scripts/train_all_subjects.sh --modality eeg --resume
./scripts/train_all_subjects.sh --modality meg --resume
./scripts/train_all_subjects.sh --modality fmri --resume
```

Pairwise shared conversion models:

```bash
./scripts/train_all_subjects.sh --modality eeg --shared-only --shared-manifest data/manifests/conversion_pools/eeg_meg.txt --resume
./scripts/train_all_subjects.sh --modality meg --shared-only --shared-manifest data/manifests/conversion_pools/eeg_meg.txt --resume

./scripts/train_all_subjects.sh --modality eeg --shared-only --shared-manifest data/manifests/conversion_pools/eeg_fmri.txt --resume
./scripts/train_all_subjects.sh --modality fmri --shared-only --shared-manifest data/manifests/conversion_pools/eeg_fmri.txt --resume

./scripts/train_all_subjects.sh --modality meg --shared-only --shared-manifest data/manifests/conversion_pools/fmri_meg.txt --resume
./scripts/train_all_subjects.sh --modality fmri --shared-only --shared-manifest data/manifests/conversion_pools/fmri_meg.txt --resume
```

3-way shared models for later robust testing:

```bash
./scripts/train_all_subjects.sh --modality eeg --shared-only --shared-manifest data/manifests/conversion_pools/eeg_fmri_meg.txt --resume
./scripts/train_all_subjects.sh --modality meg --shared-only --shared-manifest data/manifests/conversion_pools/eeg_fmri_meg.txt --resume
./scripts/train_all_subjects.sh --modality fmri --shared-only --shared-manifest data/manifests/conversion_pools/eeg_fmri_meg.txt --resume
```

## Evaluation

Full retrieval for all discovered checkpoints of one or more modalities:

```bash
./.venv/bin/python -m scripts.evaluate_all --modalities eeg,meg,fmri --skip-shared-suite --clean
```

Pairwise shared retrieval + conversion matrix:

```bash
./.venv/bin/python -m scripts.evaluate_all --modalities eeg,fmri --skip-full-retrieval --clean
```

That now defaults to the matching [data/manifests/conversion_pools](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/data/manifests/conversion_pools) manifest when one exists.

3-way shared retrieval + all pairwise conversions on the 3-way pool:

```bash
./.venv/bin/python -m scripts.evaluate_all --modalities eeg,meg,fmri --skip-full-retrieval --shared-manifest data/manifests/conversion_pools/eeg_fmri_meg.txt --clean
```

Single-subject retrieval:

```bash
./.venv/bin/python -m scripts.evaluate_retrieval --modality meg --subject 1 --ckpt checkpoints/meg/meg_brainalign_sub01_temporalcnn_best.pt --split test
```

Single pairwise conversion matrix:

```bash
./.venv/bin/python -m scripts.evaluate_conversion_matrix --source-modality eeg --target-modality fmri --source-subjects 1-10 --target-subjects 1-3 --source-shared-checkpoints --target-shared-checkpoints --shared-manifest data/manifests/conversion_pools/eeg_fmri.txt --split test
```

Rebuild summaries:

```bash
./.venv/bin/python -m scripts.summarize_results --results-root results --output-dir results/summary
```

## Notebooks

The notebooks are now intended as lightweight inspection and visualization helpers, not as alternate training pipelines:

- [notebooks/00_data_loader_scratchpad.ipynb](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/notebooks/00_data_loader_scratchpad.ipynb): loader sanity checks for EEG, MEG, and fMRI
- [notebooks/01_shared_image_intersection.ipynb](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/notebooks/01_shared_image_intersection.ipynb): overlap and shared conversion-pool inspection
- [notebooks/02_retrieval_evaluation.ipynb](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/notebooks/02_retrieval_evaluation.ipynb): checkpoint-based retrieval evaluation
- [notebooks/03_linear_baselines.ipynb](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/notebooks/03_linear_baselines.ipynb): ridge/CCA baselines on the current loaders
- [notebooks/04_conversion_heatmaps.ipynb](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/notebooks/04_conversion_heatmaps.ipynb): summary heatmaps from `results/summary/`
- [notebooks/05_retrieval_example_grids.ipynb](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/notebooks/05_retrieval_example_grids.ipynb): qualitative retrieval grids from saved checkpoints

## Related Repo Docs

- User-facing script summary: [scripts/README.md](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/scripts/README.md)
- Ordered training/evaluation runbook: [scripts/RUNBOOK.Rmd](/Users/thomas/Documents/Projects/brainalign_ext_meg_fmri/scripts/RUNBOOK.Rmd)
