from collections import defaultdict
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.eeg_loader import THINGSEEG2Dataset
from src.data.fmri_loader import THINGSfMRIDataset
from src.data.meg_loader import THINGSMEGDataset
from src.models.contrastive_model import BrainAlignModel
from src.models.fmri_model import fMRIAlignModel
from src.models.meg_model import MEGAlignModel


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def get_clip_cache_path(config):
    return os.path.join(config["data"]["clip_cache_dir"], "ViT-B-32.npz")


def load_clip_cache(config):
    return np.load(get_clip_cache_path(config))


def create_dataset(config, modality, split, subject=1, shared_only=False, quiet=False):
    clip_cache_path = get_clip_cache_path(config)
    shared_manifest_path = config["data"].get("shared_manifest_path")
    things_image_map_path = config["data"].get("things_image_map_path")

    if modality == "eeg":
        return THINGSEEG2Dataset(
            eeg_dir=config["data"]["eeg_dir"],
            clip_cache_path=clip_cache_path,
            split=split,
            subject=subject,
            quiet=quiet,
            shared_only=shared_only,
            shared_manifest_path=shared_manifest_path,
        )
    if modality == "meg":
        return THINGSMEGDataset(
            meg_dir=config["data"]["meg_dir"],
            clip_cache_path=clip_cache_path,
            split=split,
            subject=subject,
            shared_only=shared_only,
            shared_manifest_path=shared_manifest_path,
            things_image_map_path=things_image_map_path,
            quiet=quiet,
        )
    if modality == "fmri":
        return THINGSfMRIDataset(
            fmri_dir=config["data"]["fmri_dir"],
            clip_cache_path=clip_cache_path,
            split=split,
            subject=subject,
            shared_only=shared_only,
            shared_manifest_path=shared_manifest_path,
            quiet=quiet,
        )

    raise ValueError(f"Unknown modality: {modality}")


def create_dataloader(
    config,
    modality,
    split,
    subject=1,
    shared_only=False,
    quiet=False,
    shuffle=False,
):
    dataset = create_dataset(
        config,
        modality,
        split,
        subject=subject,
        shared_only=shared_only,
        quiet=quiet,
    )
    batch_size = config["training"]["batch_size"][modality]
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_model(config, modality, sample_x, device):
    if modality == "fmri":
        model = fMRIAlignModel(
            n_voxels=sample_x.shape[0],
            clip_dim=512,
            tau_init=config["model"]["temperature_init"],
        )
    elif modality == "meg":
        in_channels, seq_len = sample_x.shape
        model = MEGAlignModel(
            in_channels=in_channels,
            seq_len=seq_len,
            clip_dim=512,
            hidden_dim=config["model"].get("meg_hidden_dim", 256),
            dropout=config["model"].get("meg_dropout", 0.2),
            tau_init=config["model"]["temperature_init"],
        )
    else:
        in_channels, seq_len = sample_x.shape
        model = BrainAlignModel(
            in_channels=in_channels,
            seq_len=seq_len,
            brain_embed_dim=config["model"]["projection_dim"],
            clip_dim=512,
            tau_init=config["model"]["temperature_init"],
            modality=modality,
        )

    return model.to(device)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return checkpoint


def prepare_brain_batch(x_brain, device):
    x_brain = x_brain.to(device)
    if x_brain.dim() == 2:
        x_brain = x_brain.unsqueeze(1)
    return x_brain


def _l2_normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return matrix / norms


def collect_modality_embeddings(model, data_loader, device):
    model.eval()
    grouped_embeddings = defaultdict(list)

    with torch.no_grad():
        for batch in data_loader:
            x_brain = prepare_brain_batch(batch["x"], device)
            predictions = model(x_brain).detach().cpu().numpy().astype(np.float32)

            for idx, image_id in enumerate(batch["image_id"]):
                grouped_embeddings[str(image_id)].append(predictions[idx])

    averaged_embeddings = {}
    for image_id, vectors in grouped_embeddings.items():
        avg_vector = np.mean(np.stack(vectors, axis=0), axis=0, dtype=np.float32)
        norm = np.linalg.norm(avg_vector)
        if norm <= 0:
            raise ValueError(f"Encountered zero-norm embedding for image_id '{image_id}'")
        averaged_embeddings[image_id] = (avg_vector / norm).astype(np.float32)

    return averaged_embeddings


def stack_embeddings_by_id(embedding_dict, image_ids):
    return np.stack([embedding_dict[image_id] for image_id in image_ids], axis=0).astype(np.float32)


def clip_embeddings_for_ids(clip_dict, image_ids):
    clip_matrix = np.stack([clip_dict[image_id] for image_id in image_ids], axis=0).astype(np.float32)
    return _l2_normalize_rows(clip_matrix)


def align_embedding_dicts(source_embeddings, target_embeddings):
    image_ids = sorted(set(source_embeddings) & set(target_embeddings))
    if not image_ids:
        raise ValueError("No overlapping image IDs found between the two embedding sets")

    source_matrix = stack_embeddings_by_id(source_embeddings, image_ids)
    target_matrix = stack_embeddings_by_id(target_embeddings, image_ids)
    return image_ids, source_matrix, target_matrix


def compute_retrieval_metrics(similarity_matrix):
    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("Similarity matrix must be square after image-ID alignment")

    candidate_count = similarity_matrix.shape[0]
    if candidate_count == 0:
        return {
            "top1": 0.0,
            "top5": 0.0,
            "two_way": 0.0,
            "mrr": 0.0,
            "mean_rank": 0.0,
            "median_rank": 0.0,
            "count": 0,
        }

    labels = np.arange(candidate_count)
    ranking = np.argsort(similarity_matrix, axis=1)[:, ::-1]
    top_k = min(5, candidate_count)
    rank_positions = np.argmax(ranking == labels[:, None], axis=1)
    ranks = rank_positions + 1

    top1 = float((ranking[:, 0] == labels).mean() * 100.0)
    top5 = float(np.any(ranking[:, :top_k] == labels[:, None], axis=1).mean() * 100.0)
    mrr = float(np.mean(1.0 / ranks) * 100.0)
    mean_rank = float(np.mean(ranks))
    median_rank = float(np.median(ranks))

    if candidate_count == 1:
        two_way = 0.0
    else:
        true_similarities = np.diag(similarity_matrix)
        wins = (true_similarities[:, None] > similarity_matrix).sum(axis=1)
        two_way = float((wins / (candidate_count - 1)).mean() * 100.0)

    return {
        "top1": top1,
        "top5": top5,
        "two_way": two_way,
        "mrr": mrr,
        "mean_rank": mean_rank,
        "median_rank": median_rank,
        "count": candidate_count,
    }


def compute_bidirectional_metrics(source_matrix, target_matrix):
    similarity_matrix = source_matrix @ target_matrix.T
    return {
        "forward": compute_retrieval_metrics(similarity_matrix),
        "reverse": compute_retrieval_metrics(similarity_matrix.T),
        "candidate_count": similarity_matrix.shape[0],
    }
