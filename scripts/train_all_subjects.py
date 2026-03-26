import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.checkpoints import checkpoint_paths_for
from src.evaluate_table import main as evaluate_eeg_table_main
from src.train import train


VALID_MODALITIES = ("eeg", "meg", "fmri")


def discover_subjects(modality):
    if modality == "eeg":
        return sorted(
            int(path.name.replace("sub-", ""))
            for path in (ROOT / "data" / "things-eeg2" / "preprocessed").glob("sub-*")
            if path.is_dir()
        )

    if modality == "meg":
        preprocessed_dir = ROOT / "data" / "things-meg-ds004212" / "derivatives" / "preprocessed"
        subject_ids = set()
        for path in preprocessed_dir.glob("preprocessed_P*-epo*.fif"):
            name = path.name
            if "-epo" not in name:
                continue
            prefix = name.split("-epo", 1)[0]
            subject_ids.add(int(prefix.replace("preprocessed_P", "")))
        return sorted(subject_ids)

    if modality == "fmri":
        return sorted(
            int(path.name.replace("sub-", ""))
            for path in (ROOT / "data" / "things-fmri-ds004192" / "derivatives" / "ICA-betas").glob("sub-*")
            if path.is_dir()
        )

    raise ValueError(f"Unknown modality '{modality}'")


def main():
    parser = argparse.ArgumentParser(description="Sequential multi-subject trainer")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--modality", type=str, required=True, choices=VALID_MODALITIES)
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--resume", action="store_true", help="Resume existing checkpoints")
    parser.add_argument("--resume-best", action="store_true", help="Resume from best checkpoints instead of latest")
    parser.add_argument("--shared-only", action="store_true", help="Train conversion checkpoints on a shared manifest")
    parser.add_argument(
        "--shared-manifest",
        type=str,
        default=None,
        help="Shared image manifest required for conversion checkpoint training",
    )
    args = parser.parse_args()

    if args.shared_only and not args.shared_manifest:
        parser.error("--shared-manifest is required when --shared-only is enabled")

    subjects = discover_subjects(args.modality)
    if not subjects:
        raise SystemExit(f"No local subjects found for modality '{args.modality}'")

    print(f"Starting sequential multi-subject training for THINGS-{args.modality.upper()}...")
    for index, subject in enumerate(subjects, start=1):
        print("============================================================")
        print(f"Training Subject {subject} / {len(subjects)} ({args.modality.upper()})")
        print("============================================================")

        checkpoint_paths = checkpoint_paths_for(
            args.modality,
            subject,
            shared_only=args.shared_only,
            shared_manifest_path=args.shared_manifest,
        )
        if checkpoint_paths["best"].exists() and not args.resume and not args.resume_best:
            print(f"Checkpoint already exists at {checkpoint_paths['best']}. Skipping subject {subject}.")
            print("")
            continue

        train(
            args.config,
            args.modality,
            subject,
            epochs_override=args.epochs,
            resume=args.resume,
            resume_best=args.resume_best,
            shared_only=args.shared_only,
            shared_manifest_path=args.shared_manifest,
        )
        print(f"Finished training subject {subject}.")
        print("")

    print("============================================================")
    print(f"All discovered {args.modality.upper()} subjects have been trained.")
    print("============================================================")

    if args.modality == "eeg" and not args.shared_only:
        print("Automatically generating the EEG summary table...")
        evaluate_eeg_table_main(subject=None)
    else:
        print(f"Skipping automatic summary for {args.modality.upper()}.")
        print("Use retrieval or conversion evaluators as needed.")


if __name__ == "__main__":
    main()
