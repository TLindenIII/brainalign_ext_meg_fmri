import argparse
import torch

from src.checkpoints import (
    conversion_directory_name,
    conversion_results_path,
    evaluation_scope_for,
)
from src.eval_utils import (
    align_embedding_dicts,
    build_model,
    collect_modality_embeddings,
    compute_bidirectional_metrics,
    create_dataloader,
    load_checkpoint,
    load_config,
)
from src.data.image_manifest import (
    default_conversion_pool_manifest_path,
    default_intersection_manifest_path,
)


def build_loaded_model_and_loader(config, modality, checkpoint_path, subject, split, shared_only, device):
    data_loader = create_dataloader(
        config,
        modality,
        split,
        subject=subject,
        shared_only=shared_only,
        quiet=False,
        shuffle=False,
    )
    sample_x = data_loader.dataset[0]["x"]
    model = build_model(config, modality, sample_x, device)
    load_checkpoint(model, checkpoint_path, device)
    return model, data_loader


def build_result_lines(
    source_modality,
    source_subject,
    source_ckpt,
    target_modality,
    target_subject,
    target_ckpt,
    split,
    evaluation_scope,
    shared_group,
    aligned_count,
    metrics,
):
    forward_key = f"{source_modality}_to_{target_modality}"
    reverse_key = f"{target_modality}_to_{source_modality}"

    return [
        f"--- Conversion Results ({source_modality.upper()} sub-{source_subject:02d} <-> "
        f"{target_modality.upper()} sub-{target_subject:02d}) ---",
        f"Source checkpoint: {source_ckpt}",
        f"Target checkpoint: {target_ckpt}",
        f"Split: {split}",
        f"Evaluation scope: {evaluation_scope}",
        f"Shared group: {shared_group}",
        "Shared-only images: True",
        f"Aligned shared test images: {aligned_count}",
        "",
        forward_key,
        f"Top-1 Retrieval: {metrics['forward']['top1']:.2f}%",
        f"Top-5 Retrieval: {metrics['forward']['top5']:.2f}%",
        f"CLIP 2-Way:      {metrics['forward']['two_way']:.2f}%",
        "",
        reverse_key,
        f"Top-1 Retrieval: {metrics['reverse']['top1']:.2f}%",
        f"Top-5 Retrieval: {metrics['reverse']['top5']:.2f}%",
        f"CLIP 2-Way:      {metrics['reverse']['two_way']:.2f}%",
    ]


def write_result_lines(
    lines,
    source_modality,
    source_subject,
    target_modality,
    target_subject,
    split,
    evaluation_scope,
    shared_group,
):
    out_path = conversion_results_path(
        source_modality,
        source_subject,
        target_modality,
        target_subject,
        split,
        evaluation_scope,
        shared_group,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        handle.write("\n".join(lines))
    return out_path


def main(
    config_path,
    source_modality,
    source_ckpt,
    source_subject,
    target_modality,
    target_ckpt,
    target_subject,
    split,
    shared_manifest_path=None,
):
    config = load_config(config_path)
    if shared_manifest_path:
        config.setdefault("data", {})["shared_manifest_path"] = shared_manifest_path
    else:
        inferred_manifest = default_conversion_pool_manifest_path(config, [source_modality, target_modality])
        if not inferred_manifest.exists():
            inferred_manifest = default_intersection_manifest_path(config, [source_modality, target_modality])
        if inferred_manifest.exists():
            config.setdefault("data", {})["shared_manifest_path"] = str(inferred_manifest)
    resolved_shared_manifest_path = config.get("data", {}).get("shared_manifest_path")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(
        f"Using device: {device} for conversion evaluation "
        f"{source_modality.upper()}(sub-{source_subject:02d}) <-> "
        f"{target_modality.upper()}(sub-{target_subject:02d})"
    )

    source_model, source_loader = build_loaded_model_and_loader(
        config,
        source_modality,
        source_ckpt,
        source_subject,
        split,
        shared_only=True,
        device=device,
    )
    target_model, target_loader = build_loaded_model_and_loader(
        config,
        target_modality,
        target_ckpt,
        target_subject,
        split,
        shared_only=True,
        device=device,
    )

    print("Collecting averaged modality embeddings...")
    source_embeddings = collect_modality_embeddings(source_model, source_loader, device)
    target_embeddings = collect_modality_embeddings(target_model, target_loader, device)
    image_ids, source_matrix, target_matrix = align_embedding_dicts(source_embeddings, target_embeddings)
    metrics = compute_bidirectional_metrics(source_matrix, target_matrix)
    evaluation_scope = (
        evaluation_scope_for(shared_manifest_path=resolved_shared_manifest_path)
        if resolved_shared_manifest_path
        else "shared"
    )
    shared_group = (
        conversion_directory_name(shared_manifest_path=resolved_shared_manifest_path)
        if resolved_shared_manifest_path
        else "none"
    )

    lines = build_result_lines(
        source_modality,
        source_subject,
        source_ckpt,
        target_modality,
        target_subject,
        target_ckpt,
        split,
        evaluation_scope,
        shared_group,
        len(image_ids),
        metrics,
    )

    print("\n".join(lines))

    out_path = write_result_lines(
        lines,
        source_modality,
        source_subject,
        target_modality,
        target_subject,
        split,
        evaluation_scope,
        shared_group,
    )
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cross-modality conversion on shared images")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--source-modality", type=str, required=True, choices=["eeg", "meg", "fmri"])
    parser.add_argument("--source-ckpt", type=str, required=True, help="Checkpoint for the source modality")
    parser.add_argument("--source-subject", type=int, default=1, help="Source subject ID")
    parser.add_argument("--target-modality", type=str, required=True, choices=["eeg", "meg", "fmri"])
    parser.add_argument("--target-ckpt", type=str, required=True, help="Checkpoint for the target modality")
    parser.add_argument("--target-subject", type=int, default=1, help="Target subject ID")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--shared-manifest",
        type=str,
        default=None,
        help="Optional shared image manifest. Defaults to data/manifests/intersections/<modalities>.txt if present.",
    )
    args = parser.parse_args()

    main(
        args.config,
        args.source_modality,
        args.source_ckpt,
        args.source_subject,
        args.target_modality,
        args.target_ckpt,
        args.target_subject,
        args.split,
        args.shared_manifest,
    )
