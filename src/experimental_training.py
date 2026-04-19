"""Shared experimental training utilities for conversion Experiments 2 and 3.

These helpers intentionally write through the existing experiment-aware
checkpoint helpers. That keeps new methods under checkpoints/experiments/
without changing baseline training or evaluation paths.
"""

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.checkpoints import checkpoint_paths_for
from src.eval_utils import (
    build_model,
    compute_bidirectional_metrics,
    create_dataset,
    load_config,
    prepare_brain_batch,
)
from src.models.loss import alignment_loss, clip_loss


VALID_EXPERIMENTAL_CLIP_OBJECTIVES = ("brain_to_clip", "symmetric")


def parse_subject_spec(spec):
    subjects = set()
    for chunk in str(spec).split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid subject range '{token}'")
            subjects.update(range(start, end + 1))
        else:
            subjects.add(int(token))
    if not subjects:
        raise ValueError("No subjects parsed from subject specification")
    return sorted(subjects)


def parse_modality_subjects(value):
    parsed = {}
    for chunk in str(value).split(","):
        token = chunk.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                "Subject mappings must use '<modality>:<subject>', "
                f"got '{token}'"
            )
        modality, subject_text = token.split(":", 1)
        modality = modality.strip().lower()
        if modality in parsed:
            raise ValueError(f"Duplicate modality in subject mapping: {modality}")
        parsed[modality] = int(subject_text.strip())
    if len(parsed) < 2:
        raise ValueError("At least two modality:subject entries are required")
    return parsed


def _dataset_image_index(dataset):
    indexed = defaultdict(list)

    if hasattr(dataset, "trials"):
        for idx, trial in enumerate(dataset.trials):
            indexed[str(trial["image_id"])].append(idx)
        return indexed

    if hasattr(dataset, "files") and hasattr(dataset, "n_reps"):
        for cond_idx, image_file in enumerate(dataset.files):
            image_id = Path(image_file).stem
            start = cond_idx * dataset.n_reps
            for rep_offset in range(dataset.n_reps):
                indexed[image_id].append(start + rep_offset)
        return indexed

    for idx in range(len(dataset)):
        sample = dataset[idx]
        indexed[str(sample["image_id"])].append(idx)
    return indexed


class MultiModalImageDataset(Dataset):
    """Dataset wrapper that aligns modality datasets by shared image_id."""

    def __init__(self, modality_datasets, random_trials=False):
        self.modality_datasets = dict(modality_datasets)
        self.random_trials = random_trials
        self.image_indices = {
            modality: _dataset_image_index(dataset)
            for modality, dataset in self.modality_datasets.items()
        }
        image_sets = [set(indexed) for indexed in self.image_indices.values()]
        self.image_ids = sorted(set.intersection(*image_sets))
        if not self.image_ids:
            modalities = ", ".join(sorted(self.modality_datasets))
            raise ValueError(f"No shared image IDs found across modalities: {modalities}")

    def __len__(self):
        return len(self.image_ids)

    def _sample_index(self, indices):
        if not self.random_trials or len(indices) == 1:
            return indices[0]
        choice = torch.randint(len(indices), size=()).item()
        return indices[choice]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        x_by_modality = {}
        y_clip = None

        for modality, dataset in self.modality_datasets.items():
            sample_idx = self._sample_index(self.image_indices[modality][image_id])
            sample = dataset[sample_idx]
            x_by_modality[modality] = sample["x"]
            if y_clip is None:
                y_clip = sample["y_clip"]

        return {
            "image_id": image_id,
            "x": x_by_modality,
            "y_clip": y_clip,
        }


def create_multimodal_loader(
    config,
    modality_subjects,
    split,
    shared_manifest_path,
    batch_size=None,
    shuffle=False,
    random_trials=False,
    quiet=False,
):
    config = dict(config)
    config.setdefault("data", {})
    config["data"] = dict(config["data"])
    config["data"]["shared_manifest_path"] = shared_manifest_path

    modality_datasets = {
        modality: create_dataset(
            config,
            modality,
            split,
            subject=subject,
            shared_only=True,
            quiet=quiet,
        )
        for modality, subject in modality_subjects.items()
    }
    dataset = MultiModalImageDataset(modality_datasets, random_trials=random_trials)
    if batch_size is None:
        batch_size = min(config["training"]["batch_size"][modality] for modality in modality_subjects)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def build_models(config, train_loader, modality_subjects, device):
    first_batch = next(iter(train_loader))
    models = {}
    for modality in modality_subjects:
        sample_x = first_batch["x"][modality][0]
        models[modality] = build_model(config, modality, sample_x, device)
    return models


def _mean_logit_scale(models, modalities):
    return torch.stack([models[modality].logit_scale for modality in modalities]).mean()


def _model_parameters(models):
    for model in models.values():
        yield from model.parameters()


def _checkpoint_for_epoch(paths, resume_best=False):
    if resume_best and paths["best"].exists():
        return paths["best"]
    if paths["latest"].exists():
        return paths["latest"]
    if paths["best"].exists():
        return paths["best"]
    return None


def load_experiment_checkpoints(models, optimizer, checkpoint_paths, device, resume=False, resume_best=False):
    if not resume and not resume_best:
        return 0, 0.0

    restored_epochs = []
    best_metrics = []
    optimizer_state = None

    for modality, model in models.items():
        target = _checkpoint_for_epoch(checkpoint_paths[modality], resume_best=resume_best)
        if target is None:
            continue
        checkpoint = torch.load(target, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        if isinstance(checkpoint, dict):
            restored_epochs.append(int(checkpoint.get("epoch", -1)))
            best_metrics.append(float(checkpoint.get("best_val_metric", 0.0)))
            optimizer_state = optimizer_state or checkpoint.get("optimizer_state_dict")
        print(f"Restored {modality.upper()} checkpoint: {target}")

    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("Restored joint optimizer state.")
        except ValueError:
            print("Joint optimizer state did not match this run; continuing with a fresh optimizer.")

    start_epoch = min(restored_epochs) + 1 if restored_epochs else 0
    best_val_metric = max(best_metrics) if best_metrics else 0.0
    return start_epoch, best_val_metric


def save_experiment_checkpoints(
    models,
    optimizer,
    checkpoint_paths,
    epoch,
    best_val_metric,
    metadata,
    save_best=False,
):
    for modality, model in models.items():
        paths = checkpoint_paths[modality]
        paths["save_dir"].mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_metric": best_val_metric,
            **metadata,
        }
        torch.save(payload, paths["latest"])
        if save_best:
            torch.save(payload, paths["best"])


def collect_multimodal_embeddings(models, data_loader, device):
    for model in models.values():
        model.eval()

    grouped = {
        modality: defaultdict(list)
        for modality in models
    }

    with torch.no_grad():
        for batch in data_loader:
            for modality, model in models.items():
                x_brain = prepare_brain_batch(batch["x"][modality], device)
                predictions = model(x_brain).detach().cpu().numpy().astype(np.float32)
                for idx, image_id in enumerate(batch["image_id"]):
                    grouped[modality][str(image_id)].append(predictions[idx])

    averaged = {}
    for modality, image_map in grouped.items():
        averaged[modality] = {}
        for image_id, vectors in image_map.items():
            avg_vector = np.mean(np.stack(vectors, axis=0), axis=0, dtype=np.float32)
            norm = np.linalg.norm(avg_vector)
            if norm <= 0:
                raise ValueError(f"Encountered zero-norm embedding for {modality}:{image_id}")
            averaged[modality][image_id] = (avg_vector / norm).astype(np.float32)
    return averaged


def _stack_for_image_ids(embedding_map, image_ids):
    return np.stack([embedding_map[image_id] for image_id in image_ids], axis=0).astype(np.float32)


def evaluate_cross_modal(models, data_loader, device):
    embeddings = collect_multimodal_embeddings(models, data_loader, device)
    pair_metrics = {}

    for left, right in combinations(sorted(models), 2):
        image_ids = sorted(set(embeddings[left]) & set(embeddings[right]))
        if not image_ids:
            raise ValueError(f"No overlapping validation image IDs for {left} and {right}")
        left_matrix = _stack_for_image_ids(embeddings[left], image_ids)
        right_matrix = _stack_for_image_ids(embeddings[right], image_ids)
        pair_metrics[(left, right)] = compute_bidirectional_metrics(left_matrix, right_matrix)

    mrr_values = []
    two_way_values = []
    for metrics in pair_metrics.values():
        mrr_values.extend([metrics["forward"]["mrr"], metrics["reverse"]["mrr"]])
        two_way_values.extend([metrics["forward"]["two_way"], metrics["reverse"]["two_way"]])

    return {
        "pair_metrics": pair_metrics,
        "mean_mrr": float(np.mean(mrr_values)),
        "mean_two_way": float(np.mean(two_way_values)),
    }


def compute_joint_loss(models, batch, device, clip_objective, lambda_cross):
    y_clip = batch["y_clip"].to(device)
    predictions = {}
    clip_losses = []

    for modality, model in models.items():
        x_brain = prepare_brain_batch(batch["x"][modality], device)
        predictions[modality] = model(x_brain)
        clip_losses.append(
            alignment_loss(
                predictions[modality],
                y_clip,
                model.logit_scale.to(device),
                objective=clip_objective,
            )
        )

    cross_losses = []
    for left, right in combinations(sorted(models), 2):
        cross_losses.append(
            clip_loss(
                predictions[left],
                predictions[right],
                _mean_logit_scale(models, [left, right]).to(device),
            )
        )

    clip_anchor = torch.stack(clip_losses).mean()
    cross_modal = torch.stack(cross_losses).mean() if cross_losses else torch.zeros((), device=device)
    return clip_anchor + lambda_cross * cross_modal, clip_anchor.detach(), cross_modal.detach()


def train_joint_models(
    config_path,
    modality_subjects,
    shared_manifest_path,
    experiment_name,
    epochs_override=None,
    resume=False,
    resume_best=False,
    clip_objective="brain_to_clip",
    lambda_cross=1.0,
    batch_size=None,
    objective_name="joint_clip_xmodal",
):
    if not experiment_name:
        raise ValueError("Joint experimental training requires --experiment-name")
    if clip_objective not in VALID_EXPERIMENTAL_CLIP_OBJECTIVES:
        raise ValueError(f"Unsupported clip objective: {clip_objective}")

    config = load_config(config_path)
    config.setdefault("data", {})["shared_manifest_path"] = shared_manifest_path
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    modalities = sorted(modality_subjects)
    print(f"Using device: {device}")
    print(f"Experiment: {experiment_name}")
    print(f"Modalities: {', '.join(f'{m}:sub-{modality_subjects[m]:02d}' for m in modalities)}")
    print(f"Shared manifest: {shared_manifest_path}")
    print(f"Objective: {objective_name} | clip={clip_objective} | lambda_cross={lambda_cross}")

    train_loader = create_multimodal_loader(
        config,
        modality_subjects,
        "train",
        shared_manifest_path,
        batch_size=batch_size,
        shuffle=True,
        random_trials=True,
    )
    val_loader = create_multimodal_loader(
        config,
        modality_subjects,
        "val",
        shared_manifest_path,
        batch_size=batch_size,
        shuffle=False,
        random_trials=False,
        quiet=True,
    )
    print(f"Aligned train images: {len(train_loader.dataset)}")
    print(f"Aligned val images: {len(val_loader.dataset)}")

    models = build_models(config, train_loader, modality_subjects, device)
    optimizer = torch.optim.AdamW(
        list(_model_parameters(models)),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=1e-3,
    )
    checkpoint_paths = {
        modality: checkpoint_paths_for(
            modality,
            subject,
            shared_only=True,
            shared_manifest_path=shared_manifest_path,
            experiment_name=experiment_name,
        )
        for modality, subject in modality_subjects.items()
    }
    start_epoch, best_val_metric = load_experiment_checkpoints(
        models,
        optimizer,
        checkpoint_paths,
        device,
        resume=resume,
        resume_best=resume_best,
    )
    epochs = epochs_override
    if epochs is None:
        epochs = max(config["training"]["epochs"][modality] for modality in modalities)

    metadata = {
        "experiment_name": experiment_name,
        "alignment_objective": objective_name,
        "clip_objective": clip_objective,
        "lambda_cross": lambda_cross,
        "shared_manifest_path": shared_manifest_path,
        "modality_subjects": dict(modality_subjects),
    }

    for epoch in range(start_epoch, epochs):
        for model in models.values():
            model.train()
        total_loss = 0.0
        total_clip = 0.0
        total_cross = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            loss, clip_anchor, cross_modal = compute_joint_loss(
                models,
                batch,
                device,
                clip_objective=clip_objective,
                lambda_cross=lambda_cross,
            )
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                pbar.set_postfix({"Loss": "NaN - skipped"})
                continue
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_clip += clip_anchor.item()
            total_cross += cross_modal.item()
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "CLIP": f"{clip_anchor.item():.4f}",
                    "XModal": f"{cross_modal.item():.4f}",
                }
            )

        denom = max(1, len(train_loader))
        print(
            f"Epoch {epoch + 1} finished. "
            f"Loss: {total_loss / denom:.4f} | "
            f"CLIP: {total_clip / denom:.4f} | "
            f"XModal: {total_cross / denom:.4f}"
        )

        val_metrics = evaluate_cross_modal(models, val_loader, device)
        current_metric = val_metrics["mean_mrr"]
        print(
            f"--> Val Epoch {epoch + 1} | "
            f"Mean cross-modal MRR: {current_metric:.2f}% | "
            f"Mean cross-modal 2-way: {val_metrics['mean_two_way']:.2f}%"
        )

        save_best = current_metric > best_val_metric
        if save_best:
            best_val_metric = current_metric
            print(f"    New best cross-modal MRR: {best_val_metric:.2f}%")

        save_experiment_checkpoints(
            models,
            optimizer,
            checkpoint_paths,
            epoch,
            best_val_metric,
            metadata,
            save_best=save_best,
        )

    print(f"Training completed. Best validation metric: {best_val_metric:.2f}%")
