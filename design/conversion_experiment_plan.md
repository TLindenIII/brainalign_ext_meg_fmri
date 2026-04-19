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

| Experiment                                   | Status      | What it tests                                                                                                                                   | Main change                                                                                |
| -------------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| 1. Asymmetric brain-to-CLIP                  | Implemented | Whether removing the reverse image-to-brain term makes independently trained modality embeddings more compatible for conversion.                | Use `--alignment-objective brain_to_clip` with isolated experiment checkpoints.            |
| 2. CLIP anchor plus cross-modal loss         | Implemented | Whether conversion improves when same-image embeddings from two modalities are explicitly pulled together while still staying anchored to CLIP. | Use paired-modality training with a CLIP anchor and cross-modal contrastive term.          |
| 3. Joint multi-modality shared-pool training | Implemented | Whether training all modalities together on the same shared pool gives the most consistent conversion geometry.                                 | Train multiple modality encoders in one coordinated loop with CLIP and cross-modal losses. |

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

If your virtualenv is activated, use the normal module form:

```powershell
python -m scripts.<script_name>
```

If the virtualenv is not activated, use the venv Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m scripts.<script_name>
```

On macOS/Linux, the same direct form is:

```bash
.venv/bin/python -m scripts.<script_name>
```

The examples below use PowerShell continuation backticks. In bash/zsh, replace each trailing backtick with `\`.

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

```powershell
python -m scripts.train_all_subjects `
  --modality eeg `
  --shared-only `
  --shared-manifest data/manifests/conversion_pools/eeg_meg.txt `
  --experiment-name brain-to-clip-v1 `
  --alignment-objective brain_to_clip `
  --resume
```

Train all MEG subjects for the same pool:

```powershell
python -m scripts.train_all_subjects `
  --modality meg `
  --shared-only `
  --shared-manifest data/manifests/conversion_pools/eeg_meg.txt `
  --experiment-name brain-to-clip-v1 `
  --alignment-objective brain_to_clip `
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

Objective name:

```text
clip_anchor_xmodal
```

Implemented objective:

```text
loss = mean_m clip_alignment_loss(modality_m, clip)
     + lambda_cross * mean_pairs cross_modal_clip_loss(modality_i, modality_j)
```

The `clip_alignment_loss` defaults to `brain_to_clip`, so each modality is pulled toward the frozen CLIP embedding for the same image. The cross-modal term is a symmetric CLIP-style contrastive loss between source and target modality embeddings for aligned `image_id` batches.

Expected checkpoint namespace:

```text
--experiment-name clip-anchor-xmodal-v1
```

Design note:

- This is probably a stronger conversion objective than asymmetric CLIP-only training because it makes same-image cross-modal agreement explicit.
- It should still retain CLIP anchoring so the embedding space remains interpretable and comparable to retrieval.
- The paired trainer saves standard per-modality shared checkpoints, so existing experiment-aware evaluators can load the outputs without a new resolver.

### Training One Subject

Train EEG subject 1 and MEG subject 1 together on the EEG-MEG shared pool:

```powershell
python -m scripts.train_paired_conversion `
  --source-modality eeg `
  --target-modality meg `
  --source-subject 1 `
  --target-subject 1 `
  --shared-manifest data/manifests/conversion_pools/eeg_meg.txt `
  --experiment-name clip-anchor-xmodal-v1 `
  --alignment-objective clip_anchor_xmodal `
  --clip-objective brain_to_clip `
  --lambda-cross 1.0 `
  --resume
```

The trainer builds same-`image_id` batches from both modality datasets, computes the CLIP anchor terms, then computes the cross-modal source-target term in the same optimization step.

### Training All Subjects

Train zip-aligned EEG-MEG subject sets. This example trains EEG 1 with MEG 1, EEG 2 with MEG 2, etc.

```powershell
python -m scripts.train_paired_conversion_matrix `
  --source-modality eeg `
  --target-modality meg `
  --source-subjects 1-4 `
  --target-subjects 1-4 `
  --pairing zip `
  --shared-manifest data/manifests/conversion_pools/eeg_meg.txt `
  --experiment-name clip-anchor-xmodal-v1 `
  --alignment-objective clip_anchor_xmodal `
  --clip-objective brain_to_clip `
  --lambda-cross 1.0 `
  --resume
```

Safety note:

- `--pairing zip` is the default because it preserves one checkpoint per modality/subject.
- `--pairing product` is intentionally blocked unless `--allow-shared-checkpoint-overwrite` is provided. A Cartesian product would retrain the same subject checkpoint multiple times and the last pair would overwrite earlier pair-specific geometry.

### Evaluating Conversion

Evaluation can reuse the current experiment-aware evaluator once the checkpoints are written under the standard experiment layout:

```powershell
python -m scripts.evaluate_all `
  --modalities eeg,meg `
  --skip-full-retrieval `
  --shared-manifest data/manifests/conversion_pools/eeg_meg.txt `
  --experiment-name clip-anchor-xmodal-v1 `
  --clean
```

## Experiment 3: Joint Multi-Modality Shared-Pool Training

Objective name:

```text
joint_clip_xmodal
```

Implemented structure:

- One training loop loads paired batches by `image_id` from all modalities in the shared pool.
- Each modality keeps its own encoder.
- The training step computes CLIP anchoring and all available cross-modal positives.
- Validation computes all pairwise cross-modal retrieval metrics across the jointly trained encoders.

Expected checkpoint namespace:

```text
--experiment-name joint-shared-pool-v1
```

Design note:

- This is the cleanest way to train for conversion directly.
- It is also the strictest shared-pool test because every modality encoder is updated from the same image batch geometry.
- It writes standard per-modality shared checkpoints under the experiment namespace, keeping evaluation compatible with existing scripts.

### Training One Subject Set

Train one explicit aligned subject set:

```powershell
python -m scripts.train_joint_shared_pool `
  --modalities eeg,meg,fmri `
  --subjects eeg:1,meg:1,fmri:1 `
  --shared-manifest data/manifests/conversion_pools/eeg_fmri_meg.txt `
  --experiment-name joint-shared-pool-v1 `
  --alignment-objective joint_clip_xmodal `
  --clip-objective brain_to_clip `
  --lambda-cross 1.0 `
  --resume
```

Because subject counts differ by modality, this trainer does not assume subject IDs are naturally paired across EEG, MEG, and fMRI. The `--subjects` argument is the explicit mapping for one joint subject set.

### Training All Subjects

Train all discoverable zip-aligned subject sets:

```powershell
python -m scripts.train_joint_shared_pool `
  --modalities eeg,meg,fmri `
  --all-subjects `
  --shared-manifest data/manifests/conversion_pools/eeg_fmri_meg.txt `
  --experiment-name joint-shared-pool-v1 `
  --alignment-objective joint_clip_xmodal `
  --clip-objective brain_to_clip `
  --lambda-cross 1.0 `
  --resume
```

`--all-subjects` discovers local subjects per modality, sorts them, then trains zip-style sets by position. If counts differ, extra subjects from larger modalities are skipped instead of being trained into ambiguous checkpoints.

The trainer saves one checkpoint per modality/subject under:

```text
checkpoints/experiments/joint-shared-pool-v1/conversion/shared-eeg-fmri-meg/<modality>/
```

That keeps evaluation compatible with the existing shared-suite evaluator.

### Evaluating Conversion

Evaluate all pairwise conversions on the three-way shared pool:

```powershell
python -m scripts.evaluate_all `
  --modalities eeg,meg,fmri `
  --skip-full-retrieval `
  --shared-manifest data/manifests/conversion_pools/eeg_fmri_meg.txt `
  --experiment-name joint-shared-pool-v1 `
  --clean
```

Direct pairwise example after joint training:

```powershell
python -m scripts.evaluate_conversion_matrix `
  --source-modality eeg `
  --target-modality fmri `
  --source-subjects 1-3 `
  --target-subjects 1-3 `
  --source-shared-checkpoints `
  --target-shared-checkpoints `
  --shared-manifest data/manifests/conversion_pools/eeg_fmri_meg.txt `
  --experiment-name joint-shared-pool-v1 `
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
