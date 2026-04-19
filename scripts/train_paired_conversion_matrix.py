import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.experimental_training import (  # noqa: E402
    VALID_EXPERIMENTAL_CLIP_OBJECTIVES,
    parse_subject_spec,
    train_joint_models,
)


VALID_MODALITIES = ("eeg", "meg", "fmri")
OBJECTIVE_NAME = "clip_anchor_xmodal"


def paired_subject_sets(source_subjects, target_subjects, pairing):
    if pairing == "zip":
        if len(source_subjects) != len(target_subjects):
            raise ValueError(
                "--pairing zip requires the same number of source and target subjects "
                f"({len(source_subjects)} source vs {len(target_subjects)} target)."
            )
        return list(zip(source_subjects, target_subjects))

    return [(source, target) for source in source_subjects for target in target_subjects]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially train paired CLIP-anchor/cross-modal conversion checkpoints "
            "for multiple source-target subject sets."
        )
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--source-modality", type=str, required=True, choices=VALID_MODALITIES)
    parser.add_argument("--target-modality", type=str, required=True, choices=VALID_MODALITIES)
    parser.add_argument(
        "--source-subjects",
        type=str,
        required=True,
        help="Comma-separated subject list and/or ranges, e.g. 1-4 or 1,3,5",
    )
    parser.add_argument(
        "--target-subjects",
        type=str,
        required=True,
        help="Comma-separated subject list and/or ranges, e.g. 1-4 or 1,2",
    )
    parser.add_argument(
        "--pairing",
        type=str,
        default="zip",
        choices=["zip", "product"],
        help=(
            "zip trains subject sets by position. product trains every source-target "
            "pair and can overwrite standard per-subject checkpoints."
        ),
    )
    parser.add_argument(
        "--allow-shared-checkpoint-overwrite",
        action="store_true",
        help=(
            "Allow --pairing product even though repeated source or target subjects "
            "overwrite the same standard checkpoint names."
        ),
    )
    parser.add_argument(
        "--shared-manifest",
        type=str,
        required=True,
        help="Shared image manifest used to build aligned same-image batches",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Experiment namespace under checkpoints/experiments/<name>/",
    )
    parser.add_argument(
        "--alignment-objective",
        type=str,
        default=OBJECTIVE_NAME,
        choices=[OBJECTIVE_NAME],
        help="Named experimental objective for checkpoint metadata.",
    )
    parser.add_argument(
        "--clip-objective",
        type=str,
        default="brain_to_clip",
        choices=VALID_EXPERIMENTAL_CLIP_OBJECTIVES,
        help="CLIP-anchor loss used for each modality before the cross-modal term.",
    )
    parser.add_argument(
        "--lambda-cross",
        type=float,
        default=1.0,
        help="Weight applied to the source-target cross-modal contrastive loss.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override aligned-batch size")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoints")
    parser.add_argument("--resume-best", action="store_true", help="Resume from best checkpoints")
    args = parser.parse_args()

    if args.source_modality == args.target_modality:
        parser.error("source and target modalities must be different")
    if args.pairing == "product" and not args.allow_shared_checkpoint_overwrite:
        parser.error(
            "--pairing product would train the same subject checkpoint more than once. "
            "Use --pairing zip for isolated standard checkpoints, or add "
            "--allow-shared-checkpoint-overwrite if you intentionally want the last pair to win."
        )

    source_subjects = parse_subject_spec(args.source_subjects)
    target_subjects = parse_subject_spec(args.target_subjects)

    try:
        subject_sets = paired_subject_sets(source_subjects, target_subjects, args.pairing)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Training {len(subject_sets)} paired subject set(s) with {args.pairing} pairing.")
    for index, (source_subject, target_subject) in enumerate(subject_sets, start=1):
        print("============================================================")
        print(
            f"Pair {index}/{len(subject_sets)}: "
            f"{args.source_modality.upper()} sub-{source_subject:02d} + "
            f"{args.target_modality.upper()} sub-{target_subject:02d}"
        )
        print("============================================================")

        train_joint_models(
            args.config,
            {
                args.source_modality: source_subject,
                args.target_modality: target_subject,
            },
            args.shared_manifest,
            args.experiment_name,
            epochs_override=args.epochs,
            resume=args.resume,
            resume_best=args.resume_best,
            clip_objective=args.clip_objective,
            lambda_cross=args.lambda_cross,
            batch_size=args.batch_size,
            objective_name=args.alignment_objective,
        )


if __name__ == "__main__":
    main()
