# BrainAlign-Style Representation-First Alignment on THINGS (EEG → MEG → fMRI)

## Context

This repo implements a **representation-first, contrastive alignment pipeline** that maps brain signals into a **shared CLIP-anchored semantic space** for retrieval and cross-modal evaluation.

We will:

1. **Replicate** BrainAlign-style EEG→CLIP retrieval on **THINGS-EEG2** using **CBraMod** as the brain encoder.
2. **Extend** to MEG→CLIP retrieval on **THINGS-MEG**.
3. **Extend** to MEG↔fMRI “modality conversion” by evaluating **cross-modal retrieval** (MEG embeddings retrieve matching fMRI embeddings) in the shared CLIP space at the **image level**.
4. Compare to **linear baselines** (ridge/CCA) to test whether contrastive alignment better captures semantic structure.

---

## Goal

Test whether a **CBraMod + symmetric contrastive alignment** pipeline generalizes:

- from EEG to MEG (retrieval),
- and from MEG to fMRI (cross-modal conversion) via a shared CLIP-anchored space.

---

## Hypotheses

- **H1:** Replicating BrainAlign on THINGS-EEG2 achieves comparable Top-1/Top-5 retrieval to the paper.
- **H2:** MEG→CLIP retrieval is feasible but differs from EEG→CLIP due to sensor physics and preprocessing.
- **H3:** MEG↔fMRI conversion is measurable via retrieval in a shared CLIP-anchored space even without strict trial-level pairing.
- **H4:** Linear baselines (CCA/ridge) perform worse than contrastive alignment when semantic structure is the target.

---

## Datasets

### EEG (H1)

- **THINGS-EEG2 ([OSF](https://osf.io/3jk45/overview))**
  Download at minimum:
  - **Preprocessed EEG data**
  - **Image set**

  Optional:
  - Behavioral data (QC + metadata)

### MEG (H2, H3)

- **THINGS-MEG ([OpenNeuro ds004212](https://openneuro.org/datasets/ds004212/versions/3.0.0))**

### fMRI (H3)

- **THINGS-fMRI ([OpenNeuro ds004192](https://openneuro.org/datasets/ds004192/versions/1.0.7))**

---

## Key Design Choice: Image-Level H3

For MEG↔fMRI conversion, we evaluate at the **image level**, requiring a shared `image_id` across MEG and fMRI.
We will:

- extract stimulus identifiers from BIDS `*_events.tsv` files,
- compute the intersection of shared images,
- train separate modality→CLIP models,
- evaluate MEG→fMRI and fMRI→MEG retrieval using shared image IDs.

---

## Method Summary (BrainAlign-style)

### Anchor Space

- Use a fixed pretrained **CLIP image encoder** to embed each stimulus image:  
  `clip_img[image_id] -> y_clip ∈ R^D`
- Normalize: `y_clip = y_clip / ||y_clip||`

### Brain Encoder

- Use **CBraMod** for brain time-series encoding (EEG, MEG; and a compatible encoder for fMRI vectors if needed).
- Output: `z_brain ∈ R^{d_b}`

### Projection Head

- Map brain features to CLIP dimension:
  `p_brain = Proj(z_brain) -> R^D`
- Normalize: `p_brain = p_brain / ||p_brain||`

### Objective: Symmetric Contrastive (CLIP-style InfoNCE)

For minibatch of N matched pairs:

- Similarity: `S_ij = (p_brain_i · y_clip_j) / τ`
- Loss:
  - `L_b2i = CE(softmax(S_i,:), target=i)`
  - `L_i2b = CE(softmax(S_:,j), target=j)` (transpose)
  - `L = (L_b2i + L_i2b)/2`
- Temperature `τ` is fixed or learnable (BrainAlign uses CLIP-style logit scale initialization).

---

## Evaluation Metrics

### EEG→CLIP / MEG→CLIP Retrieval (H1/H2)

- Candidate set: held-out test images
- Compute similarity between brain embeddings and CLIP embeddings
- Report:
  - **Top-1 accuracy**
  - **Top-5 accuracy**
  - (optional) mean reciprocal rank (MRR), median rank

### MEG↔fMRI “Modality Conversion” via Cross-Modal Retrieval (H3)

On shared test images:

- Embed MEG trials -> aggregate per `image_id` -> `p_meg(image_id)`
- Embed fMRI -> aggregate per `image_id` -> `p_fmri(image_id)`
- Similarity matrix: `S = p_meg @ p_fmri.T`
- Metrics:
  - MEG→fMRI Top-1/Top-5 (correct if same `image_id`)
  - fMRI→MEG Top-1/Top-5

---

## Baselines (H4)

### Ridge Regression

- Train linear map: brain_features -> CLIP embedding
- Evaluate with same retrieval metrics.

### CCA

- Learn shared linear subspace maximizing correlation between brain features and CLIP embeddings.
- Evaluate retrieval in that shared space.

(Optionally PLS as an alternative to CCA.)

---

## Data Contract (must be consistent across modalities)

Every dataset loader should yield:

- `x`: brain data tensor
  - EEG/MEG: `C × T`
  - fMRI: `V` (voxels/ROI) or `R` features
- `image_id`: string or integer unique key for the stimulus
- `y_clip`: CLIP embedding `R^D` for `image_id`
- `meta`: subject/session/run/trial indices (optional)

### Repetitions / Aggregation

- Train on single trials where possible.
- Evaluate on **per-image aggregated embeddings** (average across trials, optionally across subjects).

---

## Implementation Plan (Tasks)

### Phase 0 — Repo Setup

- Create env (PyTorch, torchvision, CLIP, mne, numpy, pandas)
- Add config system (yaml/json)
- Define consistent logging (tensorboard/wandb optional)

### Phase 1 — CLIP Cache Builder

- Input: image directory + list of images
- Output: `clip_cache/{model_name}.npz` mapping `image_id -> embedding`
- Normalize embeddings and store float16/float32

### Phase 2 — Dataset Loaders

- EEG2 loader:
  - read preprocessed EEG arrays + trial→image mapping
  - output unified data contract
- MEG loader (BIDS):
  - parse `*_events.tsv` and locate stimulus identifiers
  - epoch data into `C×T`
- fMRI loader (BIDS):
  - obtain per-image response vectors (beta estimates or event-related aggregation)
  - parse shared `image_id`

### Phase 3 — Model Code

- CBraMod wrapper + projection head
- Train loop:
  - symmetric contrastive loss
  - periodic retrieval evaluation

### Phase 4 — Replication (H1)

- EEG2: reproduce BrainAlign-style retrieval numbers
- Track:
  - frozen vs finetuned CBraMod
  - projection dim, temperature, batch size
  - Top-1/Top-5

### Phase 5 — Extension to MEG (H2)

- Same pipeline on THINGS-MEG
- Compare retrieval performance and preprocessing effects

### Phase 6 — MEG↔fMRI Conversion (H3)

- Compute shared image intersection:
  - parse MEG `events.tsv` -> `meg_images.txt`
  - parse fMRI `events.tsv` -> `fmri_images.txt`
  - intersection -> `shared_images.txt`
- Train:
  - MEG→CLIP model
  - fMRI→CLIP model
- Evaluate cross-modal retrieval on shared test images

### Phase 7 — Linear Baselines (H4)

- Ridge and CCA baselines on each modality using identical splits
- Compare Top-1/Top-5 vs contrastive model

---

## Notes / Constraints

- EEG2 is a separate participant group from MEG/fMRI; do not require cross-participant trial pairing.
- H3 uses shared image IDs across MEG and fMRI, with conversion measured by retrieval.
- Keep splits disjoint by image_id to avoid leakage.
- Evaluate per-image aggregated embeddings for stability.
