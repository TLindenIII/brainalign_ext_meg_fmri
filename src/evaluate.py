import argparse
from pathlib import Path

import numpy as np
import torch

from src.eval_utils import (
    build_model,
    clip_embeddings_for_ids,
    collect_modality_embeddings,
    compute_bidirectional_metrics,
    create_dataloader,
    load_checkpoint,
    load_clip_cache,
    load_config,
)


def evaluate(model, test_loader, clip_dict, device):
    modality_embeddings = collect_modality_embeddings(model, test_loader, device)
    image_ids = sorted(modality_embeddings)
    modality_matrix = np.stack([modality_embeddings[image_id] for image_id in image_ids], axis=0).astype(np.float32)
    clip_matrix = clip_embeddings_for_ids(clip_dict, image_ids)

    metrics = compute_bidirectional_metrics(modality_matrix, clip_matrix)
    flat_metrics = dict(metrics["forward"])
    flat_metrics["modality_to_image"] = metrics["forward"]
    flat_metrics["image_to_modality"] = metrics["reverse"]
    flat_metrics["candidate_count"] = metrics["candidate_count"]
    return flat_metrics


def main(config_path, modality, checkpoint_path, subject, split, shared_only, shared_manifest_path=None):
    config = load_config(config_path)
    if shared_manifest_path:
        config.setdefault("data", {})["shared_manifest_path"] = shared_manifest_path
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device} to evaluate {modality.upper()} subject {subject:02d}")

    clip_dict = load_clip_cache(config)
    test_loader = create_dataloader(
        config,
        modality,
        split,
        subject=subject,
        shared_only=shared_only,
        quiet=False,
        shuffle=False,
    )

    sample_x = test_loader.dataset[0]["x"]
    model = build_model(config, modality, sample_x, device)

    print(f"Loading checkpoint: {checkpoint_path}")
    load_checkpoint(model, checkpoint_path, device)

    print("Beginning retrieval evaluation...")
    metrics = evaluate(model, test_loader, clip_dict, device)

    lines = [
        f"--- Evaluation Results ({modality.upper()} / subject {subject:02d}) ---",
        f"Checkpoint: {checkpoint_path}",
        f"Split: {split}",
        f"Shared-only images: {shared_only}",
        f"Candidate images: {metrics['candidate_count']}",
        "",
        "Modality -> Image",
        f"Top-1 Retrieval: {metrics['modality_to_image']['top1']:.2f}%",
        f"Top-5 Retrieval: {metrics['modality_to_image']['top5']:.2f}%",
        f"CLIP 2-Way:      {metrics['modality_to_image']['two_way']:.2f}%",
        "",
        "Image -> Modality",
        f"Top-1 Retrieval: {metrics['image_to_modality']['top1']:.2f}%",
        f"Top-5 Retrieval: {metrics['image_to_modality']['top5']:.2f}%",
        f"CLIP 2-Way:      {metrics['image_to_modality']['two_way']:.2f}%",
    ]

    print("\n".join(lines))

    results_dir = Path("results") / modality
    results_dir.mkdir(parents=True, exist_ok=True)
    scope = "shared" if shared_only else "full"
    out_path = results_dir / f"evaluation_sub{subject:02d}_{split}_{scope}.txt"
    with open(out_path, "w") as handle:
        handle.write("\n".join(lines))
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate modality/image retrieval against CLIP")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--modality", type=str, required=True, choices=["eeg", "meg", "fmri"])
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint to evaluate")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID to evaluate")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--shared-only",
        action="store_true",
        help="Restrict MEG/fMRI evaluation to the shared image intersection",
    )
    parser.add_argument(
        "--shared-manifest",
        type=str,
        default=None,
        help="Optional manifest of image_ids to use when --shared-only is enabled",
    )
    args = parser.parse_args()
    main(
        args.config,
        args.modality,
        args.ckpt,
        args.subject,
        args.split,
        args.shared_only,
        args.shared_manifest,
    )
