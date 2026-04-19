import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.train_all_subjects import discover_subjects  # noqa: E402
from src.experimental_training import (  # noqa: E402
    VALID_EXPERIMENTAL_CLIP_OBJECTIVES,
    parse_modality_subjects,
    train_joint_models,
)


VALID_MODALITIES = ("eeg", "meg", "fmri")
OBJECTIVE_NAME = "joint_clip_xmodal"


def parse_modalities(value):
    modalities = []
    for chunk in str(value).split(","):
        modality = chunk.strip().lower()
        if not modality:
            continue
        if modality not in VALID_MODALITIES:
            raise ValueError(f"Unknown modality '{modality}'")
        if modality in modalities:
            raise ValueError(f"Duplicate modality '{modality}'")
        modalities.append(modality)
    if len(modalities) < 2:
        raise ValueError("At least two modalities are required")
    return modalities


def all_subject_sets(modalities):
    discovered = {modality: discover_subjects(modality) for modality in modalities}
    missing = [modality for modality, subjects in discovered.items() if not subjects]
    if missing:
        raise ValueError(f"No local subjects found for: {', '.join(missing)}")

    set_count = min(len(subjects) for subjects in discovered.values())
    if set_count == 0:
        raise ValueError("No complete subject sets could be formed")

    skipped = {
        modality: subjects[set_count:]
        for modality, subjects in discovered.items()
        if len(subjects) > set_count
    }
    if skipped:
        skipped_text = ", ".join(
            f"{modality.upper()} {subjects}" for modality, subjects in skipped.items()
        )
        print(
            "Subject counts differ across modalities. "
            f"Using the first {set_count} sorted subject set(s); skipped extras: {skipped_text}"
        )

    subject_sets = []
    for index in range(set_count):
        subject_sets.append(
            {
                modality: discovered[modality][index]
                for modality in modalities
            }
        )
    return subject_sets


def explicit_subject_set(modalities, subject_mapping):
    missing = [modality for modality in modalities if modality not in subject_mapping]
    extra = [modality for modality in subject_mapping if modality not in modalities]
    if missing:
        raise ValueError(f"--subjects is missing modality entries for: {', '.join(missing)}")
    if extra:
        raise ValueError(f"--subjects includes modalities not listed in --modalities: {', '.join(extra)}")
    return [{modality: subject_mapping[modality] for modality in modalities}]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train joint shared-pool conversion models for two or more modalities "
            "with CLIP anchoring plus all pairwise cross-modal losses."
        )
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--modalities",
        type=str,
        required=True,
        help="Comma-separated modalities, e.g. eeg,meg,fmri",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Explicit subject mapping, e.g. eeg:1,meg:1,fmri:1",
    )
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="Discover local subjects and train sorted zip-style subject sets.",
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
        help="Weight applied to the mean pairwise cross-modal contrastive loss.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override aligned-batch size")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoints")
    parser.add_argument("--resume-best", action="store_true", help="Resume from best checkpoints")
    args = parser.parse_args()

    try:
        modalities = parse_modalities(args.modalities)
    except ValueError as exc:
        parser.error(str(exc))

    if args.all_subjects == bool(args.subjects):
        parser.error("Provide exactly one of --subjects or --all-subjects")

    try:
        if args.all_subjects:
            subject_sets = all_subject_sets(modalities)
        else:
            subject_sets = explicit_subject_set(
                modalities,
                parse_modality_subjects(args.subjects),
            )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Training {len(subject_sets)} joint subject set(s).")
    for index, modality_subjects in enumerate(subject_sets, start=1):
        subject_text = ", ".join(
            f"{modality.upper()} sub-{modality_subjects[modality]:02d}"
            for modality in modalities
        )
        print("============================================================")
        print(f"Joint set {index}/{len(subject_sets)}: {subject_text}")
        print("============================================================")

        train_joint_models(
            args.config,
            modality_subjects,
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
