import argparse
from pathlib import Path

import torch

from src.data.image_manifest import default_intersection_manifest_path
from src.eval_utils import (
    align_embedding_dicts,
    collect_modality_embeddings,
    compute_bidirectional_metrics,
    load_config,
)
from src.evaluate_conversion import (
    build_loaded_model_and_loader,
    build_result_lines,
    write_result_lines,
)


def parse_subject_spec(spec):
    subjects = set()
    for chunk in spec.split(","):
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


def default_checkpoint_path(modality, subject, shared_only=False):
    stem = f"{modality}_brainalign_sub{subject:02d}"
    shared_suffix = "_shared" if shared_only else ""
    base_dir = Path("checkpoints") / modality

    if modality == "meg":
        candidates = [
            base_dir / f"{stem}_temporalcnn{shared_suffix}_best.pt",
            base_dir / f"{stem}_attnpool{shared_suffix}_best.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    return base_dir / f"{stem}{shared_suffix}_best.pt"


def resolve_checkpoint_path(modality, subject, pattern=None, shared_only=False):
    if pattern:
        return Path(pattern.format(subject=subject, subject02=f"{subject:02d}"))
    return default_checkpoint_path(modality, subject, shared_only=shared_only)


def collect_subject_embeddings(
    config,
    modality,
    subjects,
    checkpoint_pattern,
    split,
    shared_manifest_path,
    shared_checkpoints,
    device,
):
    cached = {}
    for subject in subjects:
        checkpoint_path = resolve_checkpoint_path(
            modality,
            subject,
            pattern=checkpoint_pattern,
            shared_only=shared_checkpoints,
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(
            f"Loading {modality.upper()} subject {subject:02d} from {checkpoint_path} "
            f"for split '{split}'"
        )
        model, loader = build_loaded_model_and_loader(
            config,
            modality,
            str(checkpoint_path),
            subject,
            split,
            shared_only=True,
            device=device,
        )
        embeddings = collect_modality_embeddings(model, loader, device)
        cached[subject] = {
            "checkpoint_path": str(checkpoint_path),
            "embeddings": embeddings,
        }
        print(
            f"Cached {modality.upper()} subject {subject:02d}: "
            f"{len(embeddings)} aligned image embeddings"
        )
    return cached


def main(
    config_path,
    source_modality,
    target_modality,
    source_subjects_spec,
    target_subjects_spec,
    split,
    shared_manifest_path=None,
    source_ckpt_pattern=None,
    target_ckpt_pattern=None,
    source_shared_checkpoints=False,
    target_shared_checkpoints=False,
):
    config = load_config(config_path)
    if shared_manifest_path:
        config.setdefault("data", {})["shared_manifest_path"] = shared_manifest_path
    else:
        inferred_manifest = default_intersection_manifest_path(config, [source_modality, target_modality])
        if inferred_manifest.exists():
            config.setdefault("data", {})["shared_manifest_path"] = str(inferred_manifest)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    source_subjects = parse_subject_spec(source_subjects_spec)
    target_subjects = parse_subject_spec(target_subjects_spec)

    print(
        f"Using device: {device} for conversion matrix "
        f"{source_modality.upper()}[{source_subjects_spec}] <-> "
        f"{target_modality.upper()}[{target_subjects_spec}]"
    )
    print("Shared-only images: True")
    print(f"Split: {split}")
    if config.get("data", {}).get("shared_manifest_path"):
        print(f"Shared manifest: {config['data']['shared_manifest_path']}")

    source_cache = collect_subject_embeddings(
        config,
        source_modality,
        source_subjects,
        source_ckpt_pattern,
        split,
        shared_manifest_path,
        source_shared_checkpoints,
        device,
    )
    target_cache = collect_subject_embeddings(
        config,
        target_modality,
        target_subjects,
        target_ckpt_pattern,
        split,
        shared_manifest_path,
        target_shared_checkpoints,
        device,
    )

    pair_count = 0
    for source_subject in source_subjects:
        source_record = source_cache[source_subject]
        for target_subject in target_subjects:
            target_record = target_cache[target_subject]
            image_ids, source_matrix, target_matrix = align_embedding_dicts(
                source_record["embeddings"],
                target_record["embeddings"],
            )
            metrics = compute_bidirectional_metrics(source_matrix, target_matrix)
            lines = build_result_lines(
                source_modality,
                source_subject,
                source_record["checkpoint_path"],
                target_modality,
                target_subject,
                target_record["checkpoint_path"],
                split,
                len(image_ids),
                metrics,
            )
            out_path = write_result_lines(
                lines,
                source_modality,
                source_subject,
                target_modality,
                target_subject,
                split,
            )
            pair_count += 1
            print(
                f"Saved {source_modality.upper()} sub-{source_subject:02d} <-> "
                f"{target_modality.upper()} sub-{target_subject:02d} to {out_path}"
            )

    print(f"Finished {pair_count} pairwise conversion evaluations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate all pairwise subject conversions for two modalities while loading each subject once"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--source-modality", type=str, required=True, choices=["eeg", "meg", "fmri"])
    parser.add_argument("--target-modality", type=str, required=True, choices=["eeg", "meg", "fmri"])
    parser.add_argument(
        "--source-subjects",
        type=str,
        required=True,
        help="Comma-separated subject list and/or ranges, e.g. 1-10 or 1,3,5",
    )
    parser.add_argument(
        "--target-subjects",
        type=str,
        required=True,
        help="Comma-separated subject list and/or ranges, e.g. 1-4 or 1,2",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--shared-manifest",
        type=str,
        default=None,
        help="Optional shared image manifest. Defaults to data/manifests/intersections/<modalities>.txt if present.",
    )
    parser.add_argument(
        "--source-ckpt-pattern",
        type=str,
        default=None,
        help="Optional source checkpoint pattern, e.g. checkpoints/eeg/eeg_brainalign_sub{subject:02d}_best.pt",
    )
    parser.add_argument(
        "--target-ckpt-pattern",
        type=str,
        default=None,
        help="Optional target checkpoint pattern, e.g. checkpoints/meg/meg_brainalign_sub{subject:02d}_temporalcnn_best.pt",
    )
    parser.add_argument(
        "--source-shared-checkpoints",
        action="store_true",
        help="Use the default shared-only checkpoint naming for the source modality",
    )
    parser.add_argument(
        "--target-shared-checkpoints",
        action="store_true",
        help="Use the default shared-only checkpoint naming for the target modality",
    )
    args = parser.parse_args()

    main(
        args.config,
        args.source_modality,
        args.target_modality,
        args.source_subjects,
        args.target_subjects,
        args.split,
        shared_manifest_path=args.shared_manifest,
        source_ckpt_pattern=args.source_ckpt_pattern,
        target_ckpt_pattern=args.target_ckpt_pattern,
        source_shared_checkpoints=args.source_shared_checkpoints,
        target_shared_checkpoints=args.target_shared_checkpoints,
    )
