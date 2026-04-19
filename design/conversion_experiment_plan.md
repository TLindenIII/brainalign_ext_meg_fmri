# Conversion Alignment Experiment Plan

## Goal

Test whether conversion improves when shared-only conversion models are trained with objectives that more directly preserve a single frozen CLIP-aligned geometry.

The existing symmetric CLIP-style training and all existing checkpoints/results must remain untouched. Experimental runs are opt-in and are written under separate checkpoint and results roots.

## Current Baseline

The current conversion pipeline trains one subject-specific model per modality on each shared conversion pool with the symmetric CLIP / InfoNCE objective:

```text
loss = 0.5 * CE(brain @ clip.T, labels) + 0.5 * CE(clip @ brain.T, labels)
```

CLIP image embeddings are frozen. The reverse term does not move CLIP, but it does change the geometry learned by the brain encoder. Two independently trained modality encoders can both retrieve the same CLIP target while still landing in slightly different neighborhoods around that target.

## Experiment Summary

| Experiment                                   | Status                    | What it tests                                                                                                                                   | Main change                                                                                |
| -------------------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 1. Asymmetric brain-to-CLIP                  | Implemented               | Whether removing the reverse image-to-brain term makes independently trained modality embeddings more compatible for conversion.                | Use `--alignment-objective brain_to_clip` with isolated experiment checkpoints.            |
| 2. CLIP anchor plus cross-modal loss         | Proposed, not implemented | Whether conversion improves when same-image embeddings from two modalities are explicitly pulled together while still staying anchored to CLIP. | Add a paired-modality training loop and a cross-modal contrastive term.                    |
| 3. Joint multi-modality shared-pool training | Proposed, not implemented | Whether training all modalities together on the same shared pool gives the most consistent conversion geometry.                                 | Train multiple modality encoders in one coordinated loop with CLIP and cross-modal losses. |

## Experiment Isolation

Experimental checkpoints use this root:

```text
checkpoints/experiments/<experiment-name>/
```

The directory structure below that root mirrors the existing checkpoint layout:

```text
checkpoints/experiments/<experiment-name>/<modality>/
checkpoints/experiments/<experiment-name>/conversion/shared-<pool>/<modality>/
```

Experimental result files use this root:

```text
results/experiments/<experiment-name>/
```

That mirrors the current result layout:

```text
results/experiments/<experiment-name>/retrieval/...
results/experiments/<experiment-name>/conversion/...
results/experiments/<experiment-name>/summary/...
```

This keeps the current baseline paths unchanged:

```text
checkpoints/<modality>/
checkpoints/conversion/shared-<pool>/<modality>/
results/retrieval/...
results/conversion/...
results/summary/...
```

## Shared Command Variables

Examples below use EEG-MEG first because it is the strongest current conversion pair and is the best first smoke test.

Change these values for other pools:

| Pair      | Modalities     | Manifest                                           |
| --------- | -------------- | -------------------------------------------------- |
| EEG-MEG   | `eeg,meg`      | `data/manifests/conversion_pools/eeg_meg.txt`      |
| EEG-fMRI  | `eeg,fmri`     | `data/manifests/conversion_pools/eeg_fmri.txt`     |
| MEG-fMRI  | `meg,fmri`     | `data/manifests/conversion_pools/fmri_meg.txt`     |
| Three-way | `eeg,meg,fmri` | `data/manifests/conversion_pools/eeg_fmri_meg.txt` |

Use the repo virtualenv when available:

```bash
PYTHON=.venv/bin/python
```

If the virtualenv is not available, replace `$PYTHON` with the correct Python executable for the training machine.

## Experiment 1: Asymmetric Brain-to-CLIP

Objective name:

```text
brain_to_clip
```

Loss:

```text
loss = CE(brain @ clip.T, labels)
```

Purpose:

- Keep pressure focused on mapping each modality embedding toward the frozen CLIP targets.
- Remove the reverse image-to-brain retrieval term during training.
- Compare whether this produces modality embeddings that are more mutually compatible for conversion.

Safety rule:

- Any non-default alignment objective requires `--experiment-name`.
- This prevents accidentally overwriting baseline checkpoints.

### Training One Subject

Train EEG subject 1 on the EEG-MEG shared pool:

```powershell
python -m src.train `
  --modality eeg `
  --subject 1 `
  --shared-only `
  --shared-manifest data/manifests/conversion_pools/eeg_meg.txt `
  --experiment-name brain-to-clip-v1 `
  --alignment-objective brain_to_clip `
  --resume
```

Train MEG subject 1 on the same pool:

```powershell
python -m src.train `
  --modality meg `
  --subject 1 `
  --shared-only `
  --shared-manifest data/manifests/conversion_pools/eeg_meg.txt `
  --experiment-name brain-to-clip-v1 `
  --alignment-objective brain_to_clip `
  --resume
```

### Training All Subjects

Train all EEG subjects for the EEG-MEG pool:

```bash
./scripts/train_all_subjects.sh \
  --modality eeg \
  --shared-only \
  --shared-manifest data/manifests/conversion_pools/eeg_meg.txt \
  --experiment-name brain-to-clip-v1 \
  --alignment-objective brain_to_clip \
  --resume
```

Train all MEG subjects for the same pool:

```bash
./scripts/train_all_subjects.sh \
  --modality meg \
  --shared-only \
  --shared-manifest data/manifests/conversion_pools/eeg_meg.txt \
  --experiment-name brain-to-clip-v1 \
  --alignment-objective brain_to_clip \
  --resume
```

### Evaluating Conversion

Evaluate shared retrieval plus the full EEG-MEG conversion matrix for the experiment:

```powershell
python -m scripts.evaluate_all `
  --modalities eeg,meg `
  --skip-full-retrieval `
  --shared-manifest data/manifests/conversion_pools/eeg_meg.txt `
  --experiment-name brain-to-clip-v1 `
  --clean
```

Direct conversion-matrix equivalent:

```powershell
python -m scripts.evaluate_conversion_matrix `
  --source-modality eeg `
  --target-modality meg `
  --source-subjects 1-10 `
  --target-subjects 1-4 `
  --source-shared-checkpoints `
  --target-shared-checkpoints `
  --shared-manifest data/manifests/conversion_pools/eeg_meg.txt `
  --experiment-name brain-to-clip-v1 `
  --split test
```

By default this evaluates checkpoints from:

```text
checkpoints/experiments/brain-to-clip-v1/conversion/shared-eeg-meg/
```

and writes summaries to:

```text
results/experiments/brain-to-clip-v1/summary/
```

## Experiment 2: CLIP Anchor Plus Cross-Modal Loss

This is not implemented yet.

Suggested objective:

```text
loss = clip_alignment_loss(source, clip)
     + clip_alignment_loss(target, clip)
     + lambda_cross * cross_modal_loss(source, target)
```

The cross-modal term should compare embeddings for the same `image_id` across modalities. This likely needs a specialized paired-batch training script because the current trainer loads one modality at a time.

Expected checkpoint namespace:

```text
--experiment-name clip-anchor-xmodal-v1
```

Design note:

- This is probably a stronger conversion objective than asymmetric CLIP-only training because it makes same-image cross-modal agreement explicit.
- It should still retain CLIP anchoring so the embedding space remains interpretable and comparable to retrieval.

### Training One Subject

Target command shape after implementation:

```bash
$PYTHON scripts/train_paired_conversion.py \
  --source-modality eeg \
  --target-modality meg \
  --source-subject 1 \
  --target-subject 1 \
  --shared-manifest data/manifests/conversion_pools/eeg_meg.txt \
  --experiment-name clip-anchor-xmodal-v1 \
  --alignment-objective clip_anchor_xmodal \
  --lambda-cross 1.0 \
  --resume
```

This command requires a future paired trainer that can load same-`image_id` batches from both modalities in the same optimization step.

### Training All Subjects

Target command shape after implementation:

```bash
$PYTHON scripts/train_paired_conversion_matrix.py \
  --source-modality eeg \
  --target-modality meg \
  --source-subjects 1-10 \
  --target-subjects 1-4 \
  --shared-manifest data/manifests/conversion_pools/eeg_meg.txt \
  --experiment-name clip-anchor-xmodal-v1 \
  --alignment-objective clip_anchor_xmodal \
  --lambda-cross 1.0 \
  --resume
```

Implementation note: this would train pair-specific checkpoints. If we instead want one checkpoint per subject per modality, the trainer must accumulate cross-modal batches across all compatible paired subjects without overwriting the same subject checkpoint.

### Evaluating Conversion

Evaluation can reuse the current experiment-aware evaluator once the checkpoints are written under the standard experiment layout:

```bash
$PYTHON scripts/evaluate_all.py \
  --modalities eeg,meg \
  --skip-full-retrieval \
  --shared-manifest data/manifests/conversion_pools/eeg_meg.txt \
  --experiment-name clip-anchor-xmodal-v1 \
  --clean
```

If the paired trainer writes pair-specific checkpoint names instead of the standard per-modality names, evaluation will need either `--source-ckpt-pattern` and `--target-ckpt-pattern` or a small resolver update.

## Experiment 3: Joint Multi-Modality Shared-Pool Training

This is not implemented yet.

Suggested structure:

- One training loop loads paired batches by `image_id` from all modalities in the shared pool.
- Each modality keeps its own encoder.
- The training step computes CLIP anchoring and all available cross-modal positives.
- This can share temperature policy, validation logic, and early stopping across modalities.

Expected checkpoint namespace:

```text
--experiment-name joint-shared-pool-v1
```

Design note:

- This is the cleanest way to train for conversion directly.
- It is also the most invasive because it changes batching, validation, and checkpoint orchestration.

### Training One Subject Set

Target command shape after implementation for one aligned subject set:

```bash
$PYTHON scripts/train_joint_shared_pool.py \
  --modalities eeg,meg,fmri \
  --subjects eeg:1,meg:1,fmri:1 \
  --shared-manifest data/manifests/conversion_pools/eeg_fmri_meg.txt \
  --experiment-name joint-shared-pool-v1 \
  --alignment-objective joint_clip_xmodal \
  --lambda-cross 1.0 \
  --resume
```

Because subject counts differ by modality, this trainer should not assume subject IDs are naturally paired across EEG, MEG, and fMRI. The `--subjects` argument above is a proposed explicit mapping.

### Training All Subjects

Target command shape after implementation:

```bash
$PYTHON scripts/train_joint_shared_pool.py \
  --modalities eeg,meg,fmri \
  --all-subjects \
  --shared-manifest data/manifests/conversion_pools/eeg_fmri_meg.txt \
  --experiment-name joint-shared-pool-v1 \
  --alignment-objective joint_clip_xmodal \
  --lambda-cross 1.0 \
  --resume
```

Implementation note: this should probably save one checkpoint per modality/subject under:

```text
checkpoints/experiments/joint-shared-pool-v1/conversion/shared-eeg-fmri-meg/<modality>/
```

That keeps evaluation compatible with the existing shared-suite evaluator.

### Evaluating Conversion

Evaluate all pairwise conversions on the three-way shared pool:

```bash
$PYTHON scripts/evaluate_all.py \
  --modalities eeg,meg,fmri \
  --skip-full-retrieval \
  --shared-manifest data/manifests/conversion_pools/eeg_fmri_meg.txt \
  --experiment-name joint-shared-pool-v1 \
  --clean
```

Direct pairwise example after joint training:

```bash
$PYTHON scripts/evaluate_conversion_matrix.py \
  --source-modality eeg \
  --target-modality fmri \
  --source-subjects 1-10 \
  --target-subjects 1-3 \
  --source-shared-checkpoints \
  --target-shared-checkpoints \
  --shared-manifest data/manifests/conversion_pools/eeg_fmri_meg.txt \
  --experiment-name joint-shared-pool-v1 \
  --split test
```

## Comparison Checklist

For each experiment, compare against the matching baseline shared pool:

- Shared-only retrieval for each modality.
- Pairwise conversion Top-1 and Top-5.
- Pairwise conversion CLIP 2-way.
- Normalized CLIP 2-way using matched shared-pool retrieval.
- The same subject-pair count as baseline.

Recommended first test:

1. Run EEG-MEG with `brain_to_clip`; this is the strongest current conversion pair and should reveal whether the objective helps without fMRI complications.
2. If promising, repeat EEG-fMRI and MEG-fMRI.
3. Only then test the stricter three-way pool.

## Non-Goals

- Do not modify baseline checkpoint names.
- Do not overwrite `results/summary`.
- Do not change native full retrieval training unless an experiment explicitly requires it.
- Do not compare experimental conversion against baseline full retrieval only; matched shared-pool normalization is the primary reference.
