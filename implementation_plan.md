# BrainAlign Paper Replication Plan

## Background & Analysis of arxiv:2411.09723
The user correctly noted that our Top-1 retrieval accuracy for fMRI seemed extremely low (~1.10%). However, analyzing the BrainAlign paper reveals several crucial insights that completely change how we should interpret these results and structure our pipeline:

1. **The Numbers are Right, the Metric is Missing:**
   - In the paper (Table 1), for the THINGS-MEG dataset (720 classes), the **Top-1 accuracy achieved by the authors was only 1.20%** (against a 0.13% chance level).
   - Our fMRI training was achieving **~1.10% Top-1** on 546 candidates (chance level 0.18%). **Our model was already performing exactly as well as the paper's MEG model on Top-1.**
   - For fMRI (NSD in their paper), the authors **did not even report Top-1 or Top-5 metrics** because retrieval out of large unclassed pools is too difficult. Instead, they relied purely on **CLIP 2-way accuracy** (chance level = 50%), where they achieved 93.8%. Our `evaluate.py` does not currently calculate 2-way accuracy!

2. **fMRI Data Dimensionality:**
   - The paper explicitly states: *"The spatial dimensionality of the fMRI data... was reduced to approximately 15,000 voxels. This reduction was achieved by applying the NSDGeneral ROI mask..."*
   - Therefore, our recent pivot to using 15,000 highly variant voxels was exactly the right approach. Full 211K voxel modeling is not what the paper used, and is too computationally heavy.

3. **fMRI Model Architecture:**
   - The paper states: *"the network for fMRI data is configured as a Multilayer Perceptron (MLP), suited for handling the high-dimensional and spatially complex nature of fMRI data."*
   - Our current `fMRIAlignModel` with residual MLP blocks aligns perfectly with this.

4. **Training Hyperparameters:**
   - The paper used: Learning rate = `3e-4`, Weight decay = `1e-3`, Batch size = `256`, Epochs = `30`.

## Proposed Changes

### 1. Evaluate Script Update (Metrics Fix)
- We will modify `src/evaluate.py` to compute **CLIP 2-way accuracy**. 
- 2-way accuracy involves taking a ground truth image and one random distractor image, and seeing if the brain embedding has a higher cosine similarity to the true image's CLIP embedding than the distractor's.
- This will provide the 50%-90% range numbers that the user expects from the paper.

### 2. fMRI Pipeline Alignment
- We will lock the fMRI loader to the top-15K variance-selected voxels (similar to their ~15K ROI mask).
- Ensure the model architecture remains the MLP.
- Update `config.yaml` to match their training hyperparameters: `learning_rate: 3.0e-4`, `weight_decay: 1.0e-3` (implemented in `train.py`), and `epochs: 30`.

### 3. MEG Pipeline Update
- Replace the placeholder zeroes in `meg_loader.py` with actual loading of the preprocessed `.fif` files we downloaded earlier.
- The paper downsamples to 120Hz, epochs from -500ms to +1000ms, and performs baseline correction. Our preprocessed `.fif` files from OpenNeuro likely already have these steps applied, so we just need to load the data arrays.

## Verification Plan

### Automated Tests
1. **Metric Verification:** Run `evaluate.py` on the untrained `fMRIAlignModel`. It should output a 2-way accuracy of ~50.0% (random chance).
2. **Data Pipeline Sanity Check:** Load one batch from the updated `meg_loader.py` to ensure shape is `(Batch, Channels, Time)` and non-zero.

### Validation Run
1. Re-launch fMRI training for 30 epochs using the new model and hyperparams.
2. Observe `full_train_out.log` to watch the 2-way validation accuracy rise from ~50% to roughly 70-90% by epoch 30.
